function describeQatSupport(support) {
  const qatSupported = Boolean(support?.qat_supported);
  if (!qatSupported) {
    return {
      state: "unsupported",
      message: support?.qat_reason ?? "QAT is not supported for this task.",
      mode: support?.qat_mode ?? null,
      experimental: false,
    };
  }
  if (support?.qat_experimental) {
    return {
      state: "experimental",
      message: support?.qat_warning ?? "QAT is experimental for this model family.",
      mode: support?.qat_mode ?? null,
      experimental: true,
    };
  }
  return {
    state: "supported",
    message: null,
    mode: support?.qat_mode ?? null,
    experimental: false,
  };
}

function describeQatVariant(variant) {
  if (!variant?.qat || variant.qat.experimental !== true) {
    return null;
  }
  return variant.qat.warning ?? "Experimental";
}

module.exports = {
  describeQatSupport,
  describeQatVariant,
};
