const Ajv = require("ajv");
const addFormats = require("ajv-formats");
const modelConfigSchema = require("../../schemas/model-config-1.0.schema.json");

let ajvInstance = null;
const compiledByKey = new Map();
let modelConfigValidator = null;

function normalizePath(instancePath) {
  if (!instancePath) return "$";
  const dotted = String(instancePath)
    .split("/")
    .filter(Boolean)
    .map((part) => part.replace(/~1/g, "/").replace(/~0/g, "~"))
    .join(".");
  return dotted ? `$.${dotted}` : "$";
}

function getAjv() {
  if (ajvInstance) return ajvInstance;
  ajvInstance = new Ajv({
    allErrors: true,
    strict: false,
  });
  addFormats(ajvInstance);
  return ajvInstance;
}

function compileSchema(schema, cacheKey) {
  const cached = compiledByKey.get(cacheKey);
  if (cached) return cached;
  const validate = getAjv().compile(schema);
  compiledByKey.set(cacheKey, validate);
  return validate;
}

function formatAjvErrors(errors) {
  if (!Array.isArray(errors) || errors.length === 0) return [];
  return errors.map((error) => ({
    path: normalizePath(error?.instancePath),
    message: error?.message || "Invalid value",
    keyword: error?.keyword || "unknown",
  }));
}

function getModelConfigValidator() {
  if (modelConfigValidator) return modelConfigValidator;
  modelConfigValidator = compileSchema(modelConfigSchema, "model-config-1.0");
  return modelConfigValidator;
}

function validateModelConfigDraft(config) {
  const validate = getModelConfigValidator();
  const isValid = Boolean(validate(config));
  return {
    isValid,
    errors: formatAjvErrors(validate.errors),
  };
}

module.exports = {
  getAjv,
  compileSchema,
  formatAjvErrors,
  getModelConfigValidator,
  validateModelConfigDraft,
};
