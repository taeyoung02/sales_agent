import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}


/**
 * Get full path for static files
 * 
 * Cloudflare Network Terminal을 통한 서빙:
 * - NEXT_PUBLIC_STATIC_BASE_URL이 설정되면 Cloudflare 도메인 사용
 * - 설정되지 않으면 기존 방식 (Next.js rewrites) 사용
 */
export const getFullPath = (filePath: string): string => {
  // If it's already a full URL (http:// or https://), return as is
  if (filePath.startsWith('http://') || filePath.startsWith('https://')) {
    return filePath;
  }
  
  // /static/source/* 경로 처리
  if (filePath.startsWith('/static/source/')) {
    // Cloudflare 도메인이 설정되어 있으면 사용
    const staticBaseUrl = process.env.NEXT_PUBLIC_STATIC_BASE_URL;
    if (staticBaseUrl) {
      // Cloudflare를 통해 직접 접근 (CORS 문제 없음, CDN 캐싱)
      return `${staticBaseUrl}${filePath}`;
    }
    // Cloudflare 미사용 시: Next.js rewrites를 통해 프록시
    return filePath;
  }
  
  // 다른 경로는 그대로 반환 (public folder 등)
  if (filePath.startsWith('/')) {
    return filePath;
  }
  
  // 상대 경로인 경우
  return `/${filePath}`;
};