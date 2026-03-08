FROM node:20-alpine
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY src ./src
COPY next.config.js ./
COPY tsconfig.json ./
COPY next-env.d.ts ./

CMD ["npm", "run", "dev"]
