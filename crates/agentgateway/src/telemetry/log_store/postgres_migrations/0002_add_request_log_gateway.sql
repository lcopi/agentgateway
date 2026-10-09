ALTER TABLE request_logs ADD COLUMN IF NOT EXISTS gateway TEXT;
ALTER TABLE request_logs ADD COLUMN IF NOT EXISTS instance_id TEXT;

CREATE INDEX IF NOT EXISTS idx_request_logs_gateway_completed_at ON request_logs(gateway, completed_at DESC, id DESC);
