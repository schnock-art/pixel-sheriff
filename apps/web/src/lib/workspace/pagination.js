function estimateMaxVisiblePages(total, containerWidth) {
  if (total <= 0) return 0;
  if (containerWidth <= 0) return Math.min(total, 11);

  const estimatedChipWidth = 38;
  const availableForChips = Math.max(containerWidth - 28, 0);
  const chipCapacity = Math.floor(availableForChips / estimatedChipWidth);
  return Math.max(7, Math.min(total, chipCapacity));
}

function buildPageTokens(total, current, maxVisiblePages) {
  if (total <= 0) return [];
  if (total <= maxVisiblePages) {
    return Array.from({ length: total }, (_, index) => ({ type: "page", page: index + 1 }));
  }

  const currentPage = current + 1;
  const interiorBudget = Math.max(maxVisiblePages - 2, 1);
  let start = Math.max(2, currentPage - Math.floor((interiorBudget - 1) / 2));
  let end = Math.min(total - 1, start + interiorBudget - 1);
  start = Math.max(2, end - interiorBudget + 1);

  const tokens = [{ type: "page", page: 1 }];

  if (start > 2) {
    tokens.push({ type: "ellipsis", key: "left" });
  }

  for (let page = start; page <= end; page += 1) {
    tokens.push({ type: "page", page });
  }

  if (end < total - 1) {
    tokens.push({ type: "ellipsis", key: "right" });
  }

  tokens.push({ type: "page", page: total });
  return tokens;
}

module.exports = {
  estimateMaxVisiblePages,
  buildPageTokens,
};
