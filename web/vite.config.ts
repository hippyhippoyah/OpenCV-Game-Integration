import { defineConfig } from 'vitest/config';

export default defineConfig({
  // (tests run with every feature switched on: see src/test/setup.ts)
  test: { environment: 'node', include: ['src/**/*.test.ts'], setupFiles: ['src/test/setup.ts'] },
});
