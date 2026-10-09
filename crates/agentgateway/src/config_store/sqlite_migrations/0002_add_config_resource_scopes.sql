-- Every row gets a set of scopes, stored as a canonical JSON array. Existing rows become global.
-- SQLite cannot change a primary key in place, so the table is rebuilt.
CREATE TABLE agw_config_resources_scoped (
	kind TEXT NOT NULL,
	id TEXT NOT NULL,
	value_json TEXT NOT NULL CHECK (json_valid(value_json)),
	revision INTEGER NOT NULL DEFAULT 1,
	created_at TEXT NOT NULL,
	updated_at TEXT NOT NULL,
	deleted_at TEXT,
	scopes TEXT NOT NULL DEFAULT '["global"]'
		CHECK (json_valid(scopes) AND json_type(scopes) = 'array'),
	PRIMARY KEY (kind, id, scopes)
);

INSERT INTO agw_config_resources_scoped
	(kind, id, value_json, revision, created_at, updated_at, deleted_at, scopes)
SELECT kind, id, value_json, revision, created_at, updated_at, deleted_at, '["global"]'
FROM agw_config_resources;

DROP TABLE agw_config_resources;
ALTER TABLE agw_config_resources_scoped RENAME TO agw_config_resources;

CREATE INDEX IF NOT EXISTS idx_agw_config_resources_kind_updated
	ON agw_config_resources(kind, updated_at);
