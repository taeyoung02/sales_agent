import cv2
import os
import sys
import argparse

def get_blur_score(image):
    """
    이미지의 선명도를 점수로 반환합니다. (Laplacian Variance)
    높을수록 선명하고, 낮을수록 흔들린 사진입니다.
    """
    if image is None: return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def main(video_path, output_dir, interval):
    if not os.path.exists(video_path):
        print(f"Error: 파일을 찾을 수 없습니다 -> {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬 작업 시작: 총 {total_frames} 프레임 / {interval} 간격으로 추출")

    frame_idx = 0
    window_count = 0
    
    # 구간(Window) 내의 챔피언을 기억할 변수들
    best_score = -1.0
    best_frame = None
    best_frame_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break # 영상 끝
            
        # 1. 현재 프레임의 선명도 점수 계산
        score = get_blur_score(frame)
        
        # 2. 현재 구간(Interval) 내에서 점수가 더 높으면 갱신 (왕좌 탈환)
        if score > best_score:
            best_score = score
            best_frame = frame.copy() # 이미지를 메모리에 복사해둠
            best_frame_idx = frame_idx

        # 3. 구간이 끝나는 시점인지 확인 (예: 29, 59, 89...)
        # (frame_idx + 1)을 하는 이유는 0번 프레임부터 시작하기 때문
        if (frame_idx + 1) % interval == 0:
            if best_frame is not None:
                # 파일명: seq_구간번호_원본프레임번호_점수.jpg
                save_name = f"seq_{window_count:04d}_frame{best_frame_idx}_score{int(best_score)}.jpg"
                full_path = os.path.join(output_dir, save_name)
                
                # 최고 화질로 저장
                cv2.imwrite(full_path, best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                print(f"✅ 구간 {window_count} 저장: Frame {best_frame_idx} (점수: {int(best_score)})")
                
                window_count += 1
            
            # 다음 구간을 위해 변수 초기화 (리셋)
            best_score = -1.0
            best_frame = None

        frame_idx += 1

    # 4. 마지막 자투리 구간 처리 (예: 영상이 100프레임이고 간격이 30이면, 마지막 10장이 남음)
    if best_frame is not None:
        save_name = f"seq_{window_count:04d}_frame{best_frame_idx}_score{int(best_score)}.jpg"
        cv2.imwrite(os.path.join(output_dir, save_name), best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(f"✅ 마지막 구간 저장: Frame {best_frame_idx}")

    cap.release()
    print("\n✨ 모든 작업 완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='비디오에서 선명한 프레임을 추출합니다.')
    parser.add_argument('video_path', type=str, help='입력 비디오 파일 경로')
    parser.add_argument('output_dir', type=str, help='출력 디렉토리 경로')
    parser.add_argument('--interval', type=int, default=30, help='프레임 추출 간격 (기본: 30)')
    
    args = parser.parse_args()
    
    main(args.video_path, args.output_dir, args.interval)