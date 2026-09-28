"""
Custom loss function for SAM + CLIP training with background embedding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BackgroundAwareLoss(nn.Module):
    """
    Classification loss with background embedding awareness and Others penalty
    
    - 13개 vehicle part classes: Standard CrossEntropy with focal loss
    - Others class (index 13): Penalized to prevent model collapse
    """
    
    def __init__(
        self, 
        num_classes: int = 14,  # 13 vehicle parts + 1 Others
        others_class_idx: int = 13,  # "Others" is index 13
        background_weight: float = 1.0,
        temperature: float = 0.07,
        others_penalty: float = 0.5,  # Penalty multiplier for "Others" predictions
        focal_gamma: float = 2.0,  # Focal loss gamma parameter
        class_weights: torch.Tensor = None  # Optional class weights
    ):
        """
        Args:
            num_classes: Total number of classes (13 parts + 1 Others)
            others_class_idx: Index of Others/Background class (index 13)
            background_weight: Weight for background similarity loss
            temperature: Temperature for cosine similarity scaling
            others_penalty: Penalty multiplier for "Others" predictions (>1 = higher penalty)
            focal_gamma: Focal loss gamma (0 = standard CE, higher = focus on hard examples)
            class_weights: Manual class weights (optional)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.others_class_idx = others_class_idx
        self.background_weight = background_weight
        self.temperature = temperature
        self.others_penalty = others_penalty
        self.focal_gamma = focal_gamma
        
        # Class weights for imbalance handling
        self.register_buffer('class_weights', class_weights)
        
        # Standard classification loss (will be modified with focal loss)
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        else:
            self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def focal_loss(self, logits, labels):
        """
        Focal Loss implementation
        FL(p_t) = -(1 - p_t)^gamma * log(p_t)
        
        Focuses training on hard examples and down-weights easy ones.
        """
        ce_loss = self.ce_loss(logits, labels)
        
        # Get probabilities
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE) for correct class
        
        # Apply focal term: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.focal_gamma
        
        focal_loss = focal_weight * ce_loss
        
        return focal_loss
    
    def forward(
        self, 
        logits: torch.Tensor,  # (N, num_classes)
        labels: torch.Tensor,   # (N,)
        features: torch.Tensor,  # (N, clip_dim) - CLIP visual features
        background_embedding: torch.Tensor  # (1, clip_dim) - Background text embedding
    ) -> torch.Tensor:
        """
        Compute hybrid loss with Others penalty and focal loss
        
        Args:
            logits: Classification logits from model
            labels: Ground truth labels
            features: CLIP visual features (before classifier)
            background_embedding: Pre-computed background text embedding
        
        Returns:
            Total loss
        """
        # 1. Focal Loss for all samples (focuses on hard examples)
        focal_loss_values = self.focal_loss(logits, labels)
        
        # 2. Apply penalty to "Others" predictions
        # Penalize when model predicts "Others" (to prevent collapse)
        predicted_probs = torch.softmax(logits, dim=1)
        others_pred_prob = predicted_probs[:, self.others_class_idx]
        
        # Higher probability of "Others" = higher penalty
        # This discourages the model from defaulting to "Others"
        others_prediction_penalty = self.others_penalty * others_pred_prob
        
        # Combine focal loss with Others prediction penalty
        total_loss = focal_loss_values + others_prediction_penalty
        
        # 3. Additional cosine similarity loss for ground truth "Others" samples
        # (Only for samples that are truly "Others" in GT)
        others_mask = (labels == self.others_class_idx)
        
        if others_mask.sum() > 0:
            # Extract features for ground truth "Others" samples
            others_features = features[others_mask]  # (M, clip_dim)
            
            # Normalize features
            others_features_norm = F.normalize(others_features, dim=-1)
            background_norm = F.normalize(background_embedding, dim=-1)
            
            # Cosine similarity: higher is better (features should align with background)
            similarity = torch.mm(others_features_norm, background_norm.T).squeeze(-1)
            
            # Convert to loss: want to maximize similarity -> minimize negative similarity
            background_loss = -similarity / self.temperature
            
            # Apply background loss only to ground truth "Others" samples
            total_loss[others_mask] = total_loss[others_mask] + self.background_weight * background_loss
        
        # Return mean loss
        return total_loss.mean()


class ContrastiveBackgroundLoss(nn.Module):
    """
    Contrastive loss variant: push Others features towards background, away from vehicle parts
    """
    
    def __init__(
        self,
        num_classes: int = 14,
        others_class_idx: int = 13,
        margin: float = 0.5,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.others_class_idx = others_class_idx
        self.margin = margin
        self.temperature = temperature
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        background_embedding: torch.Tensor,
        part_embeddings: torch.Tensor = None  # Optional: (13, clip_dim) vehicle part text embeddings
    ) -> torch.Tensor:
        """
        Contrastive approach: 
        - Pull Others features towards background
        - Push Others features away from vehicle parts (if part_embeddings provided)
        """
        ce_loss = self.ce_loss(logits, labels)
        
        others_mask = (labels == self.others_class_idx)
        
        if others_mask.sum() > 0:
            others_features = F.normalize(features[others_mask], dim=-1)
            background_norm = F.normalize(background_embedding, dim=-1)
            
            # Positive: similarity with background
            pos_sim = torch.mm(others_features, background_norm.T).squeeze(-1)
            
            # Negative: dissimilarity with vehicle parts (if provided)
            if part_embeddings is not None:
                part_norm = F.normalize(part_embeddings, dim=-1)  # (13, clip_dim)
                neg_sim = torch.mm(others_features, part_norm.T)  # (M, 13)
                
                # Contrastive loss: maximize pos, minimize neg
                # loss = -log(exp(pos/T) / (exp(pos/T) + sum(exp(neg/T))))
                pos_exp = torch.exp(pos_sim / self.temperature)
                neg_exp = torch.exp(neg_sim / self.temperature).sum(dim=1)
                
                contrastive_loss = -torch.log(pos_exp / (pos_exp + neg_exp))
            else:
                # Simple: just push towards background
                contrastive_loss = -(pos_sim / self.temperature)
            
            ce_loss[others_mask] = ce_loss[others_mask] + contrastive_loss
        
        return ce_loss.mean()
