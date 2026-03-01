import Ajv, { type AnySchema, type ErrorObject, type ValidateFunction } from "ajv";
import addFormats from "ajv-formats";

import modelConfigSchema from "../../schemas/model-config-1.0.schema.json";

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

export function validateModelConfigDraft(config: Record<string, unknown>): { isValid: boolean; errors: SchemaIssue[] } {
  const validate = getModelConfigValidator();
  const isValid = Boolean(validate(config));
  return {
    isValid,
    errors: formatAjvErrors(validate.errors),
  };
}
