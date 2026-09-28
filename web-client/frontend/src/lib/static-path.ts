// Utility to build correct static source URLs
// Cloudflare Network Terminal을 통한 서빙 지원
// NEXT_PUBLIC_STATIC_BASE_URL이 설정되면 Cloudflare 도메인 사용
// 설정되지 않으면 기존 방식 (Next.js rewrites) 사용

export function buildPointCloudPath(vehicleId: string): string {
  const relativePath = vehicleId === "test"
    ? "/static/source/test/point_cloud.ply"
    : `/static/source/${vehicleId}/point_cloud.ply`;
  
  // Cloudflare 도메인이 설정되어 있으면 절대 URL 반환
  const staticBaseUrl = process.env.NEXT_PUBLIC_STATIC_BASE_URL;
  if (staticBaseUrl) {
    return `${staticBaseUrl}${relativePath}`;
  }
  
  // Cloudflare 미사용 시: 상대 경로 반환 (Next.js rewrites 사용)
  return relativePath;
}


