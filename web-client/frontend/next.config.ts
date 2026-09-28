import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      // /static/source/* 요청을 /api/static/source/*로 리라이트하여 CORS 우회
      // 이렇게 하면 Vercel API route를 통해 Backend로 프록시됨
      {
        source: '/static/source/:path*',
        destination: '/api/static/source/:path*',
      },
    ];
  },
};

export default nextConfig;
