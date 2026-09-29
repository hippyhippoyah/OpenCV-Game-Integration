/**
 * Switches for features that are built but not ready to show players. Flip one to true to bring
 * it back; the code behind it stays tested either way.
 */
export const FEATURES = {
  /** The Temple (embers, buildings, raids): hidden from the main menu while it's off. */
  temple: false,
  /**
   * Charged punches (a fist held at the hip or ear burns blue): too inconsistent for now. Off, fists
   * never charge, the lesson is left out of the tutorial and the campaign's practice, and the Stone
   * Garden's third flame doesn't ask for it.
   */
  chargedPunch: false,
};

/** Is this tutorial lesson switched on? (Lessons for switched-off moves are left out.) */
export const lessonOn = (id: string): boolean => id !== 'charge' || FEATURES.chargedPunch;
