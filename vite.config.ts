import { execSync } from 'node:child_process';
import { defineConfig } from 'vite';

// 画面にどの版が公開されているか表示するため、コミットとビルド日時を埋め込む
const commit = process.env.GITHUB_SHA ?? execSync('git rev-parse HEAD').toString().trim();

export default defineConfig({
  define: {
    __APP_COMMIT__: JSON.stringify(commit),
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
  },
});
