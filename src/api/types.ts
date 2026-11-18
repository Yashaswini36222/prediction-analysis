export type ModelType = 'xgboost' | 'lstm' | 'ensemble';
export type Horizon = 3 | 6 | 12;
export interface StatusResponse { status: string; data_uploaded: boolean; available_models: Record<ModelType, Horizon[]>; last_upload: string | null; stats: { uploads: number; predictions: number; trainings: number }; }
export interface UploadResponse { success: boolean; message: string; data_info: { rows: number; apps: number; categories: number; time_range: string }; next_step: string }
export interface TrainingRequest { model_type: ModelType; horizon: Horizon }
export interface TrainingStart { success: boolean; message: string; job_id: string; estimated_time: string; check_status: string; next_step: string }
export interface TrainingStatus { job_id: string; status: 'starting'|'training'|'completed'|'failed'|string; model_type: ModelType; horizon: Horizon; start_time: string; end_time?: string; progress: number }
export interface PredictionRequest { app_filter: 'all'|string[]; category_filter: 'all'|string[]; horizon?: Horizon; model_type?: ModelType; exclude_months?: string[]; exclude_apps?: string[] }
export interface PredictionsResponse { success: boolean; message: string; results: { model_used: string; horizon_months: number; future_months?: string[]; predictions: Record<string, string[]|number[]>; apps_returned?: number; total_apps_available?: number }; metadata: { prediction_time: string; processing_mode?: string; filters_applied?: unknown } }
export interface HistoricalData { apps: string[]; categories: string[]; months: string[]; rows: { app: string; digital_unit: string; category: string; values: Record<string, number> }[] }
export interface ExamplesResponse { excel_format: { expected_columns: string[]; example_row: Record<string, string|number> }; api_workflow: unknown }
export interface MetricsResponse { [key: string]: { rmse: number; mae: number; r2: number; mape: number } }
export interface ApiDetailError { detail: string }
