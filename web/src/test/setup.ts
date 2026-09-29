import { beforeEach } from 'vitest';
import { FEATURES } from '../config';

// Tests cover every feature, switched on or not in the game (see src/config.ts); a test about a
// switched-off feature turns it off itself.
beforeEach(() => {
  FEATURES.temple = true;
  FEATURES.chargedPunch = true;
});
