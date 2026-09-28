/**
 * API Proxy Route
 * 프론트엔드에서 /api/* 경로로 오는 모든 요청을 백엔드 서버로 프록시
 * 이렇게 하면 클라이언트는 상대 경로를 사용하고, 서버 측에서 Tailscale IP로 프록시됨
 */

import { NextRequest, NextResponse } from 'next/server';

// 백엔드 서버 URL (환경 변수 또는 기본값)
const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://100.93.57.71:8000';

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  const params = await context.params;
  return handleRequest(request, params, 'GET');
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  const params = await context.params;
  return handleRequest(request, params, 'POST');
}

export async function PUT(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  const params = await context.params;
  return handleRequest(request, params, 'PUT');
}

export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  const params = await context.params;
  return handleRequest(request, params, 'DELETE');
}

export async function PATCH(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  const params = await context.params;
  return handleRequest(request, params, 'PATCH');
}

export async function OPTIONS() {
  // CORS preflight 요청 처리
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, PATCH, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  });
}

async function handleRequest(
  request: NextRequest,
  params: { path: string[] },
  method: string
) {
  try {
    // 경로 재구성 (예: ['health'] -> 'health', ['chat', 'tts', 'stream-chunks'] -> 'chat/tts/stream-chunks')
    const path = params.path.join('/');
    const url = new URL(request.url);
    
    // static/source 경로는 Backend의 /static/source로 직접 프록시 (CORS 우회)
    const isStaticSource = path.startsWith('static/source/');
    let backendUrl: string;
    if (isStaticSource) {
      // /api/static/source/... -> /static/source/... (Backend의 StaticFiles mount 경로)
      // path는 이미 'static/source/...' 형태이므로 그대로 사용
      backendUrl = `${BACKEND_URL}/${path}${url.search}`;
    } else {
      // 일반 API 경로는 /api/...로 프록시
      backendUrl = `${BACKEND_URL}/api/${path}${url.search}`;
    }
    
    console.log(`[Proxy] ${method} ${path} -> ${backendUrl}${isStaticSource ? ' (static file - will stream)' : ''}`);
    
    // 요청 본문 가져오기 (POST, PUT, PATCH의 경우)
    let body: BodyInit | undefined;
    const contentType = request.headers.get('content-type');
    let isMultipartFormData = false;
    
    if (method !== 'GET' && method !== 'DELETE') {
      if (contentType?.includes('application/json')) {
        body = await request.text();
      } else if (contentType?.includes('multipart/form-data')) {
        // multipart/form-data는 FormData 객체로 전달 (boundary 포함)
        body = await request.formData();
        isMultipartFormData = true;
      } else {
        body = await request.arrayBuffer();
      }
    }
    
    // 헤더 구성
    const headers: Record<string, string> = {};
    // multipart/form-data인 경우 Content-Type 헤더를 설정하지 않음
    // FormData를 fetch로 전달할 때 브라우저가 자동으로 boundary를 포함한 Content-Type을 설정함
    if (contentType && !isMultipartFormData) {
      headers['Content-Type'] = contentType;
    }
    
    // 쿠키 전달: 브라우저에서 받은 쿠키를 백엔드로 전달 (세션 ID 포함)
    const cookieHeader = request.headers.get('cookie');
    if (cookieHeader) {
      headers['Cookie'] = cookieHeader;
    }
    
    // HTTPS 감지를 위한 프록시 헤더 전달: 백엔드의 SessionMiddleware가 HTTPS를 올바르게 감지하도록
    // Vercel은 항상 HTTPS이므로 X-Forwarded-Proto를 https로 설정
    // request.url이 https로 시작하는지도 확인
    const requestUrl = request.url;
    const isRequestHttps = requestUrl.startsWith('https://');
    const forwardedProto = request.headers.get('x-forwarded-proto') || (isRequestHttps ? 'https' : 'http');
    
    // Vercel에서는 항상 HTTPS이므로 강제로 https 설정
    headers['X-Forwarded-Proto'] = 'https';
    
    // CF-Visitor 헤더 전달 (Cloudflare를 통한 경우)
    const cfVisitor = request.headers.get('cf-visitor');
    if (cfVisitor) {
      headers['CF-Visitor'] = cfVisitor;
    } else {
      // CF-Visitor가 없어도 Vercel은 HTTPS이므로 Cloudflare 형식으로 생성
      headers['CF-Visitor'] = '{"scheme":"https"}';
    }
    
    // 디버깅: 프록시에서 백엔드로 전달하는 헤더 로깅
    console.log(`[Proxy] Forwarding headers for ${path}:`, {
      'X-Forwarded-Proto': headers['X-Forwarded-Proto'],
      'CF-Visitor': headers['CF-Visitor'],
      'Cookie': cookieHeader ? 'present' : 'absent',
    });
    
    // Range 요청 헤더 전달 (정적 파일 다운로드 최적화)
    if (isStaticSource) {
      const rangeHeader = request.headers.get('range');
      if (rangeHeader) {
        headers['Range'] = rangeHeader;
      }
    }
    
    // 백엔드로 요청 전달 (타임아웃 설정)
    // static 파일은 스트리밍이므로 타임아웃 없음
    // session-log/save는 브라우저 종료 시 호출될 수 있으므로 타임아웃을 더 길게 설정
    const isSessionLogSave = path.startsWith('chat/session-log/') && path.endsWith('/save');
    const timeout = isStaticSource 
      ? null  // static 파일은 타임아웃 없이 스트리밍
      : (isSessionLogSave ? 60000 : 30000); // session-log/save는 60초, 나머지는 30초
    
    const controller = new AbortController();
    const timeoutId = timeout ? setTimeout(() => controller.abort(), timeout) : null;
    
    try {
      const response = await fetch(backendUrl, {
        method,
        headers,
        body,
        signal: controller.signal,
      });
      
      if (timeoutId) clearTimeout(timeoutId);
      
      // 핵심: /chat/stream 및 /vanilla-chat/stream 경로는 streaming response로 처리
      const isStreamingPath = path === 'chat/stream' || path === 'vanilla-chat/stream';
      if (isStreamingPath && response.body) {
        console.log(`[Proxy] Streaming response for ${path}`);
        
        // Set-Cookie 헤더 전달: 백엔드에서 설정한 쿠키를 프론트엔드로 전달
        const streamingHeaders: Record<string, string> = {
          'Content-Type': response.headers.get('Content-Type') || 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'X-Accel-Buffering': 'no',  // Nginx buffering 방지
          // CORS 헤더
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, PATCH, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        };
        
        // Set-Cookie 헤더 전달: 백엔드에서 설정한 쿠키를 프론트엔드로 전달
        const setCookieHeader = response.headers.get('Set-Cookie');
        if (setCookieHeader) {
          streamingHeaders['Set-Cookie'] = setCookieHeader;
        }
        
        // Streaming response 직접 전달
        return new NextResponse(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: streamingHeaders,
        });
      }
      
      // static/source 파일: Range 요청 지원 및 스트리밍 (27MB 파일 최적화)
      if (isStaticSource && response.body) {
        console.log(`[Proxy] Proxying static file: ${path}`);
        
        // 모든 헤더 복사 (Content-Length 포함 - Range 요청에 필요)
        const responseHeaders: Record<string, string> = {};
        response.headers.forEach((value, key) => {
          responseHeaders[key] = value;
        });
        
        // Accept-Ranges 헤더는 항상 포함 (가이드 권장사항)
        // 브라우저가 Range 요청을 사용할 수 있음을 알림
        responseHeaders['Accept-Ranges'] = 'bytes';
        
        // Range 요청 응답인 경우 Content-Range 헤더 추가
        if (request.headers.get('range') && response.status === 206) {
          const contentRange = response.headers.get('Content-Range');
          if (contentRange) {
            responseHeaders['Content-Range'] = contentRange;
          }
        }
        
        // Streaming response 직접 전달 (Range 요청 지원)
        // 27MB 파일을 메모리에 모두 로드하지 않고 스트리밍
        return new NextResponse(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: {
            ...responseHeaders,
            // 캐시 및 CORS 헤더
            'Cache-Control': 'public, max-age=31536000, immutable',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Range',
            'Access-Control-Expose-Headers': 'Content-Length, Content-Range, Accept-Ranges',
          },
        });
      }
      
      // 일반 응답은 기존 방식대로 처리
      const data = await response.text();
      
      console.log(`[Proxy] Response status: ${response.status} for ${path}`);
      
      // 백엔드 응답의 모든 헤더 로깅 (디버깅용)
      console.log(`[Proxy] Backend response headers for ${path}:`, 
        Object.fromEntries(response.headers.entries())
      );
      
      // 응답 반환 (상태 코드, 헤더 포함)
      const responseHeaders: Record<string, string> = {
        'Content-Type': response.headers.get('Content-Type') || 'application/json',
        // CORS 헤더 추가
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, PATCH, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      };
      
      // Set-Cookie 헤더 전달: 백엔드에서 설정한 쿠키를 프론트엔드로 전달
      const setCookieHeader = response.headers.get('Set-Cookie');
      console.log(`[Proxy] Set-Cookie header from backend for ${path}:`, setCookieHeader);
      
      if (setCookieHeader) {
        responseHeaders['Set-Cookie'] = setCookieHeader;
        console.log(`[Proxy] Set-Cookie header added to response for ${path}`);
      } else {
        console.warn(`[Proxy] No Set-Cookie header in backend response for ${path}`);
      }
      
      return new NextResponse(data, {
        status: response.status,
        statusText: response.statusText,
        headers: responseHeaders,
      });
    } catch (fetchError) {
      if (timeoutId) clearTimeout(timeoutId);
      throw fetchError;
    }
  } catch (error) {
    console.error('[Proxy] Error details:', {
      error: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
      path: params.path.join('/'),
      backendUrl: BACKEND_URL,
    });
    
    return NextResponse.json(
      { 
        error: 'Failed to proxy request',
        message: error instanceof Error ? error.message : 'Unknown error',
        details: process.env.NODE_ENV === 'development' 
          ? (error instanceof Error ? error.stack : String(error))
          : undefined
      },
      { status: 500 }
    );
  }
}

