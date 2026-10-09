-- Every row gets a set of scopes. Existing rows become global. Scopes are stored sorted and
-- de-duplicated by the write path, so equal sets are equal keys.
ALTER TABLE agw_config_resources
	ADD COLUMN IF NOT EXISTS scopes TEXT[] NOT NULL DEFAULT ARRAY['global']::TEXT[];

ALTER TABLE agw_config_resources DROP CONSTRAINT IF EXISTS agw_config_resources_pkey;
ALTER TABLE agw_config_resources ADD PRIMARY KEY (kind, id, scopes);

CREATE INDEX IF NOT EXISTS idx_agw_config_resources_scopes
	ON agw_config_resources USING GIN (scopes);
