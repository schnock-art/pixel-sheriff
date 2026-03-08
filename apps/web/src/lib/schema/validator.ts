import Ajv, { type AnySchema, type ErrorObject, type ValidateFunction } from "ajv";
import addFormats from "ajv-formats";

import modelConfigSchema from "../../schemas/model-config-1.0.schema.json";
import familiesMetadata from "../metadata/families.v1.json";

export interface SchemaIssue {
  path: string;
  message: string;
  keyword: string;
}

let ajvInstance: Ajv | null = null;
const compiledByKey = new Map<string, ValidateFunction>();
let modelConfigValidator: ValidateFunction<Record<string, unknown>> | null = null;

function normalizePath(instancePath?: string): string {
  if (!instancePath) return "$";
  const dotted = instancePath
    .split("/")
    .filter(Boolean)
    .map((part) => part.replace(/~1/g, "/").replace(/~0/g, "~"))
    .join(".");
  return dotted ? `$.${dotted}` : "$";
}

export function getAjv(): Ajv {
  if (ajvInstance) return ajvInstance;
  ajvInstance = new Ajv({
    allErrors: true,
    strict: false,
  });
  addFormats(ajvInstance);
  return ajvInstance;
}

export function compileSchema<T>(schema: AnySchema, cacheKey: string): ValidateFunction<T> {
  const cached = compiledByKey.get(cacheKey);
  if (cached) return cached as ValidateFunction<T>;
  const validate = getAjv().compile<T>(schema);
  compiledByKey.set(cacheKey, validate);
  return validate;
}

export function formatAjvErrors(errors: ErrorObject[] | null | undefined): SchemaIssue[] {
  if (!errors || errors.length === 0) return [];
  return errors.map((error) => ({
    path: normalizePath(error.instancePath),
    message: error.message ?? "Invalid value",
    keyword: error.keyword,
  }));
}

export function getModelConfigValidator(): ValidateFunction<Record<string, unknown>> {
  if (modelConfigValidator) return modelConfigValidator;
  modelConfigValidator = compileSchema<Record<string, unknown>>(modelConfigSchema as AnySchema, "model-config-1.0");
  return modelConfigValidator;
}

function getFamilyInputSizeIssue(config: Record<string, unknown>): SchemaIssue | null {
  const architecture = typeof config.architecture === "object" && config.architecture !== null && !Array.isArray(config.architecture)
    ? (config.architecture as Record<string, unknown>)
    : null;
  const familyName = typeof architecture?.family === "string" ? architecture.family.trim() : "";
  if (!familyName) return null;

  const family = Array.isArray(familiesMetadata?.families)
    ? familiesMetadata.families.find((row) => row?.name === familyName)
    : null;
  const rule = family?.input_size;
  if (!rule || typeof rule !== "object") return null;

  const input = typeof config.input === "object" && config.input !== null && !Array.isArray(config.input)
    ? (config.input as Record<string, unknown>)
    : null;
  const rawSize = input?.input_size;
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

export function validateModelConfigDraft(config: Record<string, unknown>): { isValid: boolean; errors: SchemaIssue[] } {
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
