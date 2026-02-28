export interface NavigationGuardParams {
  hasUnsavedDrafts: boolean;
  confirmDiscard?: () => boolean;
}

export function shouldAllowNavigation(params: NavigationGuardParams): boolean;

