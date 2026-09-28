# frontend/Dockerfile (단일 스테이지, 일반 빌드용)
FROM node:20-alpine

# 시스템 라이브러리
RUN apk add --no-cache libc6-compat

WORKDIR /app

# Yarn 4 활성화
RUN corepack enable
RUN corepack prepare yarn@4.9.2 --activate

# 의존성 설치
COPY package.json yarn.lock* .yarnrc.yml ./
RUN yarn install --frozen-lockfile

# 애플리케이션 코드 복사
COPY . .

# 프로덕션 빌드
RUN yarn build

# 런타임 설정
ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

EXPOSE 3000

# Next.js 서버 시작 (package.json의 "start": "next start")
CMD ["yarn", "start"]