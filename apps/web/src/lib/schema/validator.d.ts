import type Ajv from "ajv";
import type { AnySchema, ErrorObject, ValidateFunction } from "ajv";

export interface SchemaIssue {
  path: string;
  message: string;
  keyword: string;
}

export function getAjv(): Ajv;
export function compileSchema<T>(schema: AnySchema, cacheKey: string): ValidateFunction<T>;
export function formatAjvErrors(errors: ErrorObject[] | null | undefined): SchemaIssue[];
export function getModelConfigValidator(): ValidateFunction<Record<string, unknown>>;
export function validateModelConfigDraft(config: Record<string, unknown>): { isValid: boolean; errors: SchemaIssue[] };
