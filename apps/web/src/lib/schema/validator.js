const Ajv = require("ajv");
const addFormats = require("ajv-formats");
const modelConfigSchema = require("../../schemas/model-config-1.0.schema.json");
const familiesMetadata = require("../metadata/families.v1.json");

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

function getFamilyInputSizeIssue(config) {
  const architecture = config && typeof config === "object" && !Array.isArray(config) ? config.architecture : null;
  if (!architecture || typeof architecture !== "object" || Array.isArray(architecture)) return null;
  const familyName = typeof architecture.family === "string" ? architecture.family.trim() : "";
  if (!familyName) return null;

  const family = Array.isArray(familiesMetadata?.families)
    ? familiesMetadata.families.find((row) => row && row.name === familyName)
    : null;
  const rule = family && typeof family === "object" ? family.input_size : null;
  if (!rule || typeof rule !== "object" || Array.isArray(rule)) return null;

  const input = config && typeof config === "object" && !Array.isArray(config) ? config.input : null;
  if (!input || typeof input !== "object" || Array.isArray(input)) return null;
  const rawSize = input.input_size;
  if (!Array.isArray(rawSize) || rawSize.length !== 2 || !rawSize.every((value) => Number.isInteger(value))) return null;
  const width = Number(rawSize[0]);
  const height = Number(rawSize[1]);
  if (width < 1 || height < 1) return null;

  if (rule.shape === "square" && width !== height) {
    return {
      path: "$.input.input_size",
      message: `${familyName} requires square input sizes`,
      keyword: "familyInputSize",
    };
  }

  if (rule.mode === "fixed") {
    const required = Number(rule.required_square_size);
    if (Number.isInteger(required) && required > 0 && width !== required) {
      return {
        path: "$.input.input_size",
        message: `${familyName} requires input_size [${required}, ${required}]`,
        keyword: "familyInputSize",
      };
    }
    return null;
  }

  if (rule.mode === "range") {
    const minimum = Number(rule.min_square_size);
    const step = Number(rule.step);
    if (Number.isInteger(minimum) && minimum > 0 && width < minimum) {
      return {
        path: "$.input.input_size",
        message: `${familyName} requires square input_size >= ${minimum}`,
        keyword: "familyInputSize",
      };
    }
    if (Number.isInteger(step) && step > 0 && Number.isInteger(minimum) && minimum > 0 && (width - minimum) % step !== 0) {
      return {
        path: "$.input.input_size",
        message: `${familyName} requires square input_size in increments of ${step} from ${minimum}`,
        keyword: "familyInputSize",
      };
    }
  }

  return null;
}

function validateModelConfigDraft(config) {
  const validate = getModelConfigValidator();
  const schemaValid = Boolean(validate(config));
  const familyIssue = getFamilyInputSizeIssue(config);
  const errors = [...formatAjvErrors(validate.errors)];
  if (familyIssue) errors.push(familyIssue);
  return {
    isValid: schemaValid && familyIssue === null,
    errors,
  };
}

module.exports = {
  getAjv,
  compileSchema,
  formatAjvErrors,
  getModelConfigValidator,
  validateModelConfigDraft,
};
