/**
 * TTS Engine using Web Speech API
 */

export interface TTSOptions {
  rate?: number;
  pitch?: number;
  volume?: number;
  voice?: SpeechSynthesisVoice;
  lang?: string;
}

export class TTSEngine {
  private synth: SpeechSynthesis;
  private voices: SpeechSynthesisVoice[] = [];
  private isVoicesLoaded: boolean = false;

  constructor() {
    if (typeof window === "undefined") {
      throw new Error("TTSEngine can only be used in browser environment");
    }

    this.synth = window.speechSynthesis;
    this.loadVoices();

    // Some browsers load voices asynchronously
    this.synth.onvoiceschanged = () => {
      this.loadVoices();
    };
  }

  private loadVoices() {
    this.voices = this.synth.getVoices();
    // Filter for Korean and English voices
    this.voices = this.voices.filter(
      (v) =>
        v.lang.startsWith("ko") ||
        v.lang.startsWith("en") 
    );
    this.isVoicesLoaded = true;
  }

  /**
   * Get available voices
   */
  getVoices(): SpeechSynthesisVoice[] {
    if (!this.isVoicesLoaded) {
      this.loadVoices();
    }
    return this.voices;
  }

  /**
   * Find best voice for given language
   */
  findVoice(lang: string = "ko"): SpeechSynthesisVoice | undefined {
    const voices = this.getVoices();
    // Try exact match first
    let voice = voices.find((v) => v.lang === lang);
    if (voice) return voice;

    // Try language prefix match
    voice = voices.find((v) => v.lang.startsWith(lang.split("-")[0]));
    if (voice) return voice;

    // Return first available voice
    return voices[0];
  }

  /**
   * Speak text with TTS
   */
  speak(text: string, options?: TTSOptions): Promise<void> {
    return new Promise((resolve, reject) => {
      // Ensure voices are loaded
      this.loadVoices();
      
      // Wait a bit for voices to load if they're not ready
      if (this.voices.length === 0) {
        console.warn("[TTS] No voices available, waiting for voices to load...");
        // Wait for voices to load (max 2 seconds)
        const startTime = Date.now();
        const checkVoices = setInterval(() => {
          this.loadVoices();
          if (this.voices.length > 0 || Date.now() - startTime > 2000) {
            clearInterval(checkVoices);
            this.attemptSpeak(text, options, resolve, reject);
          }
        }, 100);
        return;
      }

      this.attemptSpeak(text, options, resolve, reject);
    });
  }

  /**
   * Attempt to speak text (internal method)
   */
  private attemptSpeak(
    text: string,
    options: TTSOptions | undefined,
    resolve: () => void,
    reject: (error: Error) => void
  ): void {
    try {
      // Cancel any ongoing speech to prevent conflicts
      if (this.synth.speaking) {
        this.synth.cancel();
      }

      const utterance = new SpeechSynthesisUtterance(text);

      utterance.rate = options?.rate || 1.0;
      utterance.pitch = options?.pitch || 1.0;
      utterance.volume = options?.volume ?? 1.0;
      utterance.lang = options?.lang || "ko-KR";

      // Set voice
      if (options?.voice) {
        utterance.voice = options.voice;
      } else {
        const defaultVoice = this.findVoice(options?.lang || "ko");
        if (defaultVoice) {
          utterance.voice = defaultVoice;
          console.log(`[TTS] Using voice: ${defaultVoice.name} (${defaultVoice.lang})`);
        } else {
          console.warn("[TTS] No suitable voice found, using default");
        }
      }

      utterance.onstart = () => {
        console.log(`[TTS] Started speaking: "${text.substring(0, 50)}..."`);
      };

      utterance.onend = () => {
        console.log("[TTS] Speech completed");
        resolve();
      };

      utterance.onerror = (event) => {
        console.error(`[TTS] Error:`, event);
        reject(new Error(`TTS error: ${event.error}`));
      };

      // Speak with a small delay to ensure user interaction context is maintained
      setTimeout(() => {
        this.synth.speak(utterance);
      }, 10);
    } catch (error) {
      console.error("[TTS] Exception during speak:", error);
      reject(error instanceof Error ? error : new Error(String(error)));
    }
  }

  /**
   * Cancel all TTS
   */
  cancel() {
    this.synth.cancel();
  }

  /**
   * Pause TTS
   */
  pause() {
    this.synth.pause();
  }

  /**
   * Resume paused TTS
   */
  resume() {
    this.synth.resume();
  }

  /**
   * Check if TTS is speaking
   */
  isSpeaking(): boolean {
    return this.synth.speaking;
  }

  /**
   * Check if TTS is paused
   */
  isPaused(): boolean {
    return this.synth.paused;
  }
}

// Singleton instance
let ttsEngineInstance: TTSEngine | null = null;

export function getTTSEngine(): TTSEngine {
  if (typeof window === "undefined") {
    throw new Error("TTSEngine can only be used in browser environment");
  }

  if (!ttsEngineInstance) {
    ttsEngineInstance = new TTSEngine();
  }
  return ttsEngineInstance;
}

