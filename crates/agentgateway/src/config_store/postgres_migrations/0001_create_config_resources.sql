-- Schema created by releases before numbered migrations. IF NOT EXISTS keeps this a no-op on
-- databases that already have it.
CREATE TABLE IF NOT EXISTS agw_config_resources (
	kind TEXT NOT NULL,
	id TEXT NOT NULL,
	value_json JSONB NOT NULL,
	revision BIGINT NOT NULL DEFAULT 1,
	created_at TIMESTAMPTZ NOT NULL,
	updated_at TIMESTAMPTZ NOT NULL,
	deleted_at TIMESTAMPTZ,
	PRIMARY KEY (kind, id)
);

CREATE INDEX IF NOT EXISTS idx_agw_config_resources_kind_updated
	ON agw_config_resources(kind, updated_at);
