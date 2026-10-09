use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use arc_swap::ArcSwap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::postgres::PgListener;
use sqlx::types::Json;
use sqlx::{PgPool, Postgres, QueryBuilder, Row, Sqlite, SqlitePool, Transaction};
use tokio::sync::watch;
use tracing::{error, warn};

use crate::database::DatabasePool;
use crate::telemetry::log_store;

/// The default scope of a row.
pub const GLOBAL_SCOPE: &str = "global";

#[derive(Clone, Debug)]
pub struct ConfigResourceStore {
	pool: DatabasePool,
	change_tx: watch::Sender<()>,
	notification_id: Option<String>,
	/// Rows whose scopes overlap these are listed by `list` and trigger reloads.
	visible_scopes: Arc<ArcSwap<Vec<String>>>,
	write_hook: Option<Arc<dyn ConfigWriteHook>>,
}

/// Which rows a list returns, by scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeFilter {
	/// Rows overlapping the store's visible scopes.
	Visible,
	All,
	Overlapping(Vec<String>),
}

/// A row change passed to a [`ConfigWriteHook`]. `None` scopes mean the row does not exist
/// before (insert) or after (delete) the write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigWrite {
	pub kind: ConfigResourceKind,
	pub id: String,
	pub previous_scopes: Option<Vec<String>>,
	pub scopes: Option<Vec<String>>,
}

/// Runs inside each Postgres write transaction before a row changes; an error aborts the write.
#[async_trait::async_trait]
pub trait ConfigWriteHook: Send + Sync + fmt::Debug {
	async fn before_write(
		&self,
		conn: &mut sqlx::PgConnection,
		write: &ConfigWrite,
	) -> anyhow::Result<()>;
}

pub fn global_scopes() -> Vec<String> {
	vec![GLOBAL_SCOPE.to_string()]
}

/// Sorted and de-duplicated; empty means global.
pub fn canonical_scopes(scopes: impl IntoIterator<Item = impl Into<String>>) -> Vec<String> {
	let mut scopes: Vec<String> = scopes.into_iter().map(Into::into).collect();
	scopes.sort();
	scopes.dedup();
	if scopes.is_empty() {
		return global_scopes();
	}
	scopes
}

pub fn is_global_scopes(scopes: &[String]) -> bool {
	scopes.is_empty() || scopes == [GLOBAL_SCOPE]
}

pub fn scopes_overlap(left: &[String], right: &[String]) -> bool {
	left.iter().any(|scope| right.contains(scope))
}

pub fn filter_visible(resources: Vec<ConfigResource>, visible: &[String]) -> Vec<ConfigResource> {
	resources
		.into_iter()
		.filter(|resource| scopes_overlap(&resource.scopes, visible))
		.collect()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ConfigResource {
	pub kind: ConfigResourceKind,
	pub id: String,
	#[serde(default = "global_scopes")]
	pub scopes: Vec<String>,
	pub value: Value,
	pub revision: i64,
	pub created_at: DateTime<Utc>,
	pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigResourcesResponse {
	pub resources: Vec<ConfigResource>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigResourceError {
	#[error("{0}")]
	InvalidRequest(String),
	#[error("{0}")]
	Conflict(String),
	#[error("{0}")]
	NotFound(String),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ConfigResourceKind {
	#[serde(rename = "modelCatalog")]
	ModelCatalog,
	#[serde(rename = "llm.provider")]
	LlmProvider,
	#[serde(rename = "llm.model")]
	LlmModel,
	#[serde(rename = "llm.virtualModel")]
	LlmVirtualModel,
	#[serde(rename = "llm.apiKey")]
	LlmApiKey,
	#[serde(rename = "llm.policy")]
	LlmPolicy,
	#[serde(rename = "mcp.target")]
	McpTarget,
	#[serde(rename = "mcp.policy")]
	McpPolicy,
	#[serde(rename = "llm.settings")]
	LlmSettings,
	#[serde(rename = "mcp.settings")]
	McpSettings,
	#[serde(rename = "traffic.gateway")]
	TrafficGateway,
	#[serde(rename = "traffic.route")]
	TrafficRoute,
	#[serde(rename = "traffic.tcpRoute")]
	TrafficTcpRoute,
	#[serde(rename = "ui.policy")]
	UiPolicy,
	#[serde(rename = "frontend.policy")]
	FrontendPolicy,
}

impl ConfigResourceKind {
	pub(crate) fn settings_fields(self) -> Option<(&'static str, &'static [&'static str])> {
		match self {
			Self::LlmSettings => Some(("llm", &["gateways", "port", "tls"])),
			Self::McpSettings => Some(("mcp", &MCP_SETTINGS_FIELDS)),
			_ => None,
		}
	}

	pub const fn as_str(self) -> &'static str {
		match self {
			Self::ModelCatalog => "modelCatalog",
			Self::LlmProvider => "llm.provider",
			Self::LlmModel => "llm.model",
			Self::LlmVirtualModel => "llm.virtualModel",
			Self::LlmApiKey => "llm.apiKey",
			Self::LlmPolicy => "llm.policy",
			Self::McpTarget => "mcp.target",
			Self::McpPolicy => "mcp.policy",
			Self::McpSettings => "mcp.settings",
			Self::LlmSettings => "llm.settings",
			Self::TrafficGateway => "traffic.gateway",
			Self::TrafficRoute => "traffic.route",
			Self::TrafficTcpRoute => "traffic.tcpRoute",
			Self::UiPolicy => "ui.policy",
			Self::FrontendPolicy => "frontend.policy",
		}
	}
}

impl fmt::Display for ConfigResourceKind {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(self.as_str())
	}
}

impl FromStr for ConfigResourceKind {
	type Err = ConfigResourceError;

	fn from_str(kind: &str) -> Result<Self, Self::Err> {
		match kind {
			"modelCatalog" => Ok(Self::ModelCatalog),
			"llm.provider" => Ok(Self::LlmProvider),
			"llm.model" => Ok(Self::LlmModel),
			"llm.virtualModel" => Ok(Self::LlmVirtualModel),
			"llm.apiKey" => Ok(Self::LlmApiKey),
			"llm.policy" => Ok(Self::LlmPolicy),
			"mcp.target" => Ok(Self::McpTarget),
			"mcp.policy" => Ok(Self::McpPolicy),
			"mcp.settings" => Ok(Self::McpSettings),
			"llm.settings" => Ok(Self::LlmSettings),
			"traffic.gateway" => Ok(Self::TrafficGateway),
			"traffic.route" => Ok(Self::TrafficRoute),
			"traffic.tcpRoute" => Ok(Self::TrafficTcpRoute),
			"ui.policy" => Ok(Self::UiPolicy),
			"frontend.policy" => Ok(Self::FrontendPolicy),
			_ => Err(ConfigResourceError::InvalidRequest(format!(
				"unsupported config resource kind: {kind}"
			))),
		}
	}
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ConfigResourceUpsertRequest {
	#[serde(default)]
	pub resources: Vec<ConfigResourceUpsert>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigResourceUpsert {
	pub value: Value,
}

pub async fn setup(cfg: &log_store::Config) -> anyhow::Result<ConfigResourceStore> {
	ConfigResourceStore::connect(&cfg.url, cfg.max_connections).await
}

impl ConfigResourceStore {
	async fn connect(url: &str, max_connections: Option<u32>) -> anyhow::Result<Self> {
		Self::from_pool(DatabasePool::connect_with_max_connections(url, max_connections).await?).await
	}

	pub async fn from_pool(pool: DatabasePool) -> anyhow::Result<Self> {
		let (change_tx, _) = watch::channel(());
		let visible_scopes = Arc::new(ArcSwap::from_pointee(global_scopes()));
		match &pool {
			DatabasePool::Sqlite(pool) => {
				let mut migrator = sqlx::migrate!("./src/config_store/sqlite_migrations");
				migrator.dangerous_set_table_name(MIGRATIONS_TABLE);
				migrator.run(pool).await?;
				Ok(Self {
					pool: DatabasePool::Sqlite(pool.clone()),
					change_tx,
					notification_id: None,
					visible_scopes,
					write_hook: None,
				})
			},
			DatabasePool::Postgres(pool) => {
				let mut migrator = sqlx::migrate!("./src/config_store/postgres_migrations");
				migrator.dangerous_set_table_name(MIGRATIONS_TABLE);
				migrator.run(pool).await?;
				let notification_id = uuid::Uuid::new_v4().to_string();
				let mut listener = PgListener::connect_with(pool).await?;
				listener.listen(POSTGRES_CHANGE_CHANNEL).await?;
				let listener_change_tx = change_tx.clone();
				let listener_notification_id = notification_id.clone();
				let listener_visible_scopes = visible_scopes.clone();
				tokio::spawn(async move {
					loop {
						match listener.try_recv().await {
							Ok(Some(notification)) => {
								if requires_reload(
									notification.payload(),
									&listener_notification_id,
									&listener_visible_scopes.load(),
								) {
									let _ = listener_change_tx.send(());
								}
							},
							Ok(None) => {
								warn!("postgres config change listener reconnected; reloading config");
								let _ = listener_change_tx.send(());
							},
							Err(err) => {
								error!(?err, "postgres config change listener failed");
								tokio::time::sleep(std::time::Duration::from_secs(1)).await;
							},
						}
					}
				});
				Ok(Self {
					pool: DatabasePool::Postgres(pool.clone()),
					change_tx,
					notification_id: Some(notification_id),
					visible_scopes,
					write_hook: None,
				})
			},
		}
	}

	/// Installs a hook that runs inside every Postgres write transaction.
	pub fn with_write_hook(mut self, hook: Arc<dyn ConfigWriteHook>) -> Self {
		self.write_hook = Some(hook);
		self
	}

	/// Sets the scopes `list` returns and reloads follow. Defaults to `["global"]`.
	pub fn set_visible_scopes(&self, scopes: impl IntoIterator<Item = impl Into<String>>) {
		self
			.visible_scopes
			.store(Arc::new(canonical_scopes(scopes)));
	}

	/// The per-process ID sent with change notifications (Postgres only).
	pub fn notification_id(&self) -> Option<&str> {
		self.notification_id.as_deref()
	}

	pub fn pool(&self) -> DatabasePool {
		self.pool.clone()
	}

	pub fn subscribe_changes(&self) -> watch::Receiver<()> {
		self.change_tx.subscribe()
	}

	pub async fn list(
		&self,
		kind: Option<ConfigResourceKind>,
	) -> anyhow::Result<Vec<ConfigResource>> {
		self.list_scoped(kind, ScopeFilter::Visible).await
	}

	pub async fn list_scoped(
		&self,
		kind: Option<ConfigResourceKind>,
		filter: ScopeFilter,
	) -> anyhow::Result<Vec<ConfigResource>> {
		let scopes = match filter {
			ScopeFilter::Visible => Some(self.visible_scopes.load().to_vec()),
			ScopeFilter::All => None,
			ScopeFilter::Overlapping(scopes) => Some(scopes),
		};
		match &self.pool {
			DatabasePool::Sqlite(pool) => list_sqlite(pool, kind, scopes.as_deref()).await,
			DatabasePool::Postgres(pool) => list_postgres(pool, kind, scopes.as_deref()).await,
		}
	}

	/// Writes rows addressed by `(kind, id, scopes)`. A resource whose `previous_scopes` differ
	/// from `scopes` moves that row to the new scopes in place.
	pub(crate) async fn upsert_prepared(
		&self,
		prepared: Vec<PreparedResource>,
	) -> anyhow::Result<ConfigResourcesResponse> {
		for resource in &prepared {
			validate_id(&resource.id)?;
		}
		let (resources, changed_scopes) = match &self.pool {
			DatabasePool::Sqlite(pool) => upsert_sqlite(pool, prepared).await?,
			DatabasePool::Postgres(pool) => {
				upsert_postgres(
					pool,
					prepared,
					self
						.notification_id
						.as_deref()
						.expect("postgres store has a notification ID"),
					self.write_hook.as_deref(),
				)
				.await?
			},
		};
		if !resources.is_empty() {
			self.notify_changed(&changed_scopes);
		}
		Ok(ConfigResourcesResponse { resources })
	}

	pub(crate) async fn rename_prepared(
		&self,
		previous_kind: ConfigResourceKind,
		previous_id: &str,
		prepared: PreparedResource,
	) -> anyhow::Result<ConfigResourcesResponse> {
		validate_id(previous_id)?;
		validate_id(&prepared.id)?;
		let changed_scopes = canonical_scopes([prepared.target_scopes(), &prepared.scopes].concat());
		let resource = match &self.pool {
			DatabasePool::Sqlite(pool) => {
				rename_sqlite(pool, previous_kind, previous_id, prepared).await?
			},
			DatabasePool::Postgres(pool) => {
				rename_postgres(
					pool,
					previous_kind,
					previous_id,
					prepared,
					self
						.notification_id
						.as_deref()
						.expect("postgres store has a notification ID"),
					self.write_hook.as_deref(),
				)
				.await?
			},
		};
		self.notify_changed(&changed_scopes);
		Ok(ConfigResourcesResponse {
			resources: vec![resource],
		})
	}

	pub async fn delete(&self, kind: ConfigResourceKind, id: &str) -> anyhow::Result<()> {
		self.delete_scoped(kind, id, &global_scopes()).await
	}

	pub async fn delete_scoped(
		&self,
		kind: ConfigResourceKind,
		id: &str,
		scopes: &[String],
	) -> anyhow::Result<()> {
		validate_id(id)?;
		let deleted = match &self.pool {
			DatabasePool::Sqlite(pool) => delete_sqlite(pool, kind, id, scopes).await?,
			DatabasePool::Postgres(pool) => {
				delete_postgres(
					pool,
					kind,
					id,
					scopes,
					self
						.notification_id
						.as_deref()
						.expect("postgres store has a notification ID"),
					self.write_hook.as_deref(),
				)
				.await?
			},
		};
		if !deleted {
			return Err(
				ConfigResourceError::NotFound(format!("config resource not found: {kind}/{id}")).into(),
			);
		}
		self.notify_changed(scopes);
		Ok(())
	}

	fn notify_changed(&self, scopes: &[String]) {
		if scopes_overlap(scopes, &self.visible_scopes.load()) {
			let _ = self.change_tx.send(());
		}
	}
}

/// Whether a change notification from another process touched a visible scope. Payloads that are
/// not JSON are a bare origin ID with unknown scopes.
fn requires_reload(payload: &str, self_id: &str, visible: &[String]) -> bool {
	#[derive(Deserialize)]
	struct Notification {
		origin: String,
		scopes: Vec<String>,
	}
	match serde_json::from_str::<Notification>(payload) {
		Ok(n) => n.origin != self_id && scopes_overlap(&n.scopes, visible),
		Err(_) => payload != self_id,
	}
}

const MIGRATIONS_TABLE: &str = "_agentgateway_config_migrations";

fn validate_id(id: &str) -> anyhow::Result<()> {
	if id.is_empty() {
		return Err(
			ConfigResourceError::InvalidRequest("config resource id cannot be empty".to_string()).into(),
		);
	}
	Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedResource {
	pub kind: ConfigResourceKind,
	pub id: String,
	pub value: Value,
	pub scopes: Vec<String>,
	/// Scopes of an existing row to move to `scopes`.
	pub previous_scopes: Option<Vec<String>>,
}

impl PreparedResource {
	pub(crate) fn new(kind: ConfigResourceKind, id: String, value: Value) -> Self {
		Self {
			kind,
			id,
			value,
			scopes: global_scopes(),
			previous_scopes: None,
		}
	}

	/// Scopes of the row this write addresses.
	fn target_scopes(&self) -> &[String] {
		self.previous_scopes.as_deref().unwrap_or(&self.scopes)
	}
}

pub(crate) const MCP_SETTINGS_FIELDS: [&str; 5] = [
	"gateways",
	"port",
	"statefulMode",
	"prefixMode",
	"failureMode",
];

const API_KEY_METADATA_PREFIX: &str = "agentgateway.dev/";
const API_KEY_ID_METADATA: &str = "agentgateway.dev/id";
const API_KEY_CREATED_AT_METADATA: &str = "agentgateway.dev/createdAt";
const API_KEY_HINT_METADATA: &str = "agentgateway.dev/keyHint";
/// Scopes of the row that defines an API key, set on keys from non-global rows.
pub(crate) const API_KEY_SCOPES_METADATA: &str = "agentgateway.dev/scopes";

/// Older file keys have no stored ID, so expose their array position to the resource API.
fn file_api_key_id(value: &Value, index: usize) -> String {
	api_key_metadata(value)
		.and_then(|metadata| metadata.get(API_KEY_ID_METADATA))
		.and_then(Value::as_str)
		.filter(|id| !id.is_empty())
		.map(ToString::to_string)
		.unwrap_or_else(|| format!("@index:{index}"))
}

fn api_key_metadata(value: &Value) -> Option<&serde_json::Map<String, Value>> {
	value.get("metadata").and_then(Value::as_object)
}

pub(crate) fn file_api_key_created_at(config: &Value, id: &str) -> Option<i64> {
	crate::json::traverse(config, &["llm", "policies", "apiKey", "keys"])
		.and_then(Value::as_array)?
		.iter()
		.enumerate()
		.find(|(index, value)| file_api_key_id(value, *index) == id)
		.and_then(|(_, value)| api_key_created_at(value))
}

pub(crate) fn api_key_created_at(value: &Value) -> Option<i64> {
	api_key_metadata(value)?
		.get(API_KEY_CREATED_AT_METADATA)?
		.as_i64()
}

/// Direct mappings from a resource kind to its native YAML collection.
enum FileResourceCollection {
	List(&'static [&'static str]),
	Map(&'static [&'static str]),
}

fn file_resource_collection(kind: ConfigResourceKind) -> Option<FileResourceCollection> {
	match kind {
		ConfigResourceKind::LlmProvider => Some(FileResourceCollection::List(&["llm", "providers"])),
		ConfigResourceKind::LlmModel => Some(FileResourceCollection::List(&["llm", "models"])),
		ConfigResourceKind::LlmVirtualModel => {
			Some(FileResourceCollection::List(&["llm", "virtualModels"]))
		},
		ConfigResourceKind::McpTarget => Some(FileResourceCollection::List(&["mcp", "targets"])),
		ConfigResourceKind::LlmPolicy => Some(FileResourceCollection::Map(&["llm", "policies"])),
		ConfigResourceKind::McpPolicy => Some(FileResourceCollection::Map(&["mcp", "policies"])),
		ConfigResourceKind::UiPolicy => Some(FileResourceCollection::Map(&["ui", "policies"])),
		ConfigResourceKind::FrontendPolicy => Some(FileResourceCollection::Map(&["frontendPolicies"])),
		ConfigResourceKind::TrafficGateway => Some(FileResourceCollection::Map(&["gateways"])),
		ConfigResourceKind::TrafficRoute => Some(FileResourceCollection::List(&["routes"])),
		ConfigResourceKind::TrafficTcpRoute => Some(FileResourceCollection::List(&["tcpRoutes"])),
		ConfigResourceKind::ModelCatalog
		| ConfigResourceKind::LlmApiKey
		| ConfigResourceKind::McpSettings
		| ConfigResourceKind::LlmSettings => None,
	}
}

pub(crate) fn file_config_resource<'a>(
	config: &'a Value,
	kind: ConfigResourceKind,
	id: &str,
) -> Option<&'a Value> {
	if kind == ConfigResourceKind::LlmApiKey {
		return config
			.pointer("/llm/policies/apiKey/keys")?
			.as_array()?
			.iter()
			.enumerate()
			.find(|(index, value)| file_api_key_id(value, *index) == id)
			.map(|(_, value)| value);
	}
	match file_resource_collection(kind)? {
		FileResourceCollection::Map(path) => crate::json::traverse(config, path)?.get(id),
		FileResourceCollection::List(path) => crate::json::traverse(config, path)?
			.as_array()?
			.iter()
			.find(|value| resource_id(kind, value).is_ok_and(|current| current == id)),
	}
}

pub(crate) fn upsert_file_config_resource(
	config: &mut Value,
	prepared: &PreparedResource,
	previous_id: Option<&str>,
) -> anyhow::Result<()> {
	if let Some(collection) = file_resource_collection(prepared.kind) {
		return match collection {
			FileResourceCollection::List(path) => {
				upsert_file_list_resource(config, path, prepared, previous_id)
			},
			FileResourceCollection::Map(path) => {
				upsert_file_map_resource(config, path, prepared, previous_id)
			},
		};
	}
	match prepared.kind {
		ConfigResourceKind::ModelCatalog => upsert_file_model_catalog(config, &prepared.value),
		ConfigResourceKind::LlmApiKey => upsert_file_api_key(config, prepared, previous_id),
		ConfigResourceKind::McpSettings | ConfigResourceKind::LlmSettings => {
			upsert_file_settings(config, prepared.kind, &prepared.value)
		},
		ConfigResourceKind::LlmProvider
		| ConfigResourceKind::LlmModel
		| ConfigResourceKind::LlmVirtualModel
		| ConfigResourceKind::LlmPolicy
		| ConfigResourceKind::McpTarget
		| ConfigResourceKind::McpPolicy
		| ConfigResourceKind::TrafficGateway
		| ConfigResourceKind::TrafficRoute
		| ConfigResourceKind::TrafficTcpRoute
		| ConfigResourceKind::UiPolicy
		| ConfigResourceKind::FrontendPolicy => unreachable!("direct file resources handled above"),
	}
}

pub(crate) fn delete_file_config_resource(
	config: &mut Value,
	kind: ConfigResourceKind,
	id: &str,
) -> anyhow::Result<bool> {
	if let Some(collection) = file_resource_collection(kind) {
		return match collection {
			FileResourceCollection::List(path) => delete_file_list_resource(config, path, kind, id),
			FileResourceCollection::Map(path) => delete_file_map_resource(config, path, id),
		};
	}
	match kind {
		ConfigResourceKind::ModelCatalog => delete_file_model_catalog(config),
		ConfigResourceKind::LlmApiKey => delete_file_api_key(config, id),
		ConfigResourceKind::McpSettings | ConfigResourceKind::LlmSettings => {
			delete_file_settings(config, kind)
		},
		ConfigResourceKind::LlmProvider
		| ConfigResourceKind::LlmModel
		| ConfigResourceKind::LlmVirtualModel
		| ConfigResourceKind::LlmPolicy
		| ConfigResourceKind::McpTarget
		| ConfigResourceKind::McpPolicy
		| ConfigResourceKind::TrafficGateway
		| ConfigResourceKind::TrafficRoute
		| ConfigResourceKind::TrafficTcpRoute
		| ConfigResourceKind::UiPolicy
		| ConfigResourceKind::FrontendPolicy => unreachable!("direct file resources handled above"),
	}
}

/// Creates missing parent objects and the array at `path`, rejecting incompatible shapes.
fn ensure_file_array<'a>(
	config: &'a mut Value,
	path: &[&str],
) -> anyhow::Result<&'a mut Vec<Value>> {
	let (field, parents) = path
		.split_last()
		.ok_or_else(|| anyhow::anyhow!("file resource path cannot be empty"))?;
	let mut current = config;
	for parent in parents {
		let object = current
			.as_object_mut()
			.ok_or_else(|| anyhow::anyhow!("local config {} must be a JSON object", parents.join(".")))?;
		current = object
			.entry((*parent).to_string())
			.or_insert_with(|| Value::Object(serde_json::Map::new()));
	}
	current
		.as_object_mut()
		.ok_or_else(|| anyhow::anyhow!("local config {} must be a JSON object", parents.join(".")))?
		.entry((*field).to_string())
		.or_insert_with(|| Value::Array(Vec::new()))
		.as_array_mut()
		.ok_or_else(|| anyhow::anyhow!("local config {} must be an array", path.join(".")))
}

/// Creates missing objects along `path`, rejecting any existing non-object value.
fn ensure_file_object<'a>(
	config: &'a mut Value,
	path: &[&str],
) -> anyhow::Result<&'a mut serde_json::Map<String, Value>> {
	let mut current = config;
	for field in path {
		let object = current
			.as_object_mut()
			.ok_or_else(|| anyhow::anyhow!("local config {} must be a JSON object", path.join(".")))?;
		current = object
			.entry((*field).to_string())
			.or_insert_with(|| Value::Object(serde_json::Map::new()));
	}
	current
		.as_object_mut()
		.ok_or_else(|| anyhow::anyhow!("local config {} must be a JSON object", path.join(".")))
}

fn upsert_file_list_resource(
	config: &mut Value,
	path: &[&str],
	prepared: &PreparedResource,
	previous_id: Option<&str>,
) -> anyhow::Result<()> {
	let values = ensure_file_array(config, path)?;
	let existing = values.iter().position(|value| {
		resource_id(prepared.kind, value).is_ok_and(|id| id == previous_id.unwrap_or(&prepared.id))
	});
	if let Some(previous_id) = previous_id {
		let Some(existing) = existing else {
			return Err(
				ConfigResourceError::NotFound(format!(
					"config resource not found: {}/{}",
					prepared.kind, previous_id
				))
				.into(),
			);
		};
		if previous_id != prepared.id
			&& values.iter().enumerate().any(|(index, value)| {
				index != existing && resource_id(prepared.kind, value).is_ok_and(|id| id == prepared.id)
			}) {
			return Err(
				ConfigResourceError::Conflict(format!(
					"config resource already exists: {}/{}",
					prepared.kind, prepared.id
				))
				.into(),
			);
		}
		values[existing] = prepared.value.clone();
	} else if let Some(existing) = existing {
		values[existing] = prepared.value.clone();
	} else {
		values.push(prepared.value.clone());
	}
	Ok(())
}

fn delete_file_list_resource(
	config: &mut Value,
	path: &[&str],
	kind: ConfigResourceKind,
	id: &str,
) -> anyhow::Result<bool> {
	let Some(values) = crate::json::traverse_mut(config, path).and_then(Value::as_array_mut) else {
		return Ok(false);
	};
	let before = values.len();
	values.retain(|value| resource_id(kind, value).map_or(true, |resource_id| resource_id != id));
	Ok(values.len() != before)
}

fn upsert_file_map_resource(
	config: &mut Value,
	path: &[&str],
	prepared: &PreparedResource,
	previous_id: Option<&str>,
) -> anyhow::Result<()> {
	let values = ensure_file_object(config, path)?;
	if let Some(previous_id) = previous_id {
		let policy_upsert = previous_id == prepared.id
			&& matches!(
				prepared.kind,
				ConfigResourceKind::LlmPolicy
					| ConfigResourceKind::McpPolicy
					| ConfigResourceKind::UiPolicy
					| ConfigResourceKind::FrontendPolicy
			);
		if !values.contains_key(previous_id) && !policy_upsert {
			return Err(
				ConfigResourceError::NotFound(format!(
					"config resource not found: {}/{}",
					prepared.kind, previous_id
				))
				.into(),
			);
		}
		if previous_id != prepared.id && values.contains_key(&prepared.id) {
			return Err(
				ConfigResourceError::Conflict(format!(
					"config resource already exists: {}/{}",
					prepared.kind, prepared.id
				))
				.into(),
			);
		}
		if previous_id != prepared.id {
			values.remove(previous_id);
		}
	}
	let mut value = prepared.value.clone();
	// API keys are managed as separate resources and must survive policy updates.
	if prepared.kind == ConfigResourceKind::LlmPolicy && prepared.id == "apiKey" {
		let existing_keys = values
			.get("apiKey")
			.and_then(|policy| policy.get("keys"))
			.cloned();
		value
			.as_object_mut()
			.ok_or_else(|| anyhow::anyhow!("llm.policy/apiKey must be an object"))?
			.insert(
				"keys".to_string(),
				existing_keys.unwrap_or_else(|| Value::Array(Vec::new())),
			);
	}
	// A gateway's resource ID is the key in the YAML map, not a field in its value.
	if prepared.kind == ConfigResourceKind::TrafficGateway {
		value
			.as_object_mut()
			.ok_or_else(|| anyhow::anyhow!("traffic.gateway/{} must be an object", prepared.id))?
			.remove("name");
	}
	values.insert(prepared.id.clone(), value);
	Ok(())
}

fn delete_file_map_resource(config: &mut Value, path: &[&str], id: &str) -> anyhow::Result<bool> {
	Ok(
		crate::json::traverse_mut(config, path)
			.and_then(Value::as_object_mut)
			.and_then(|values| values.remove(id))
			.is_some(),
	)
}

fn upsert_file_api_key(
	config: &mut Value,
	prepared: &PreparedResource,
	previous_id: Option<&str>,
) -> anyhow::Result<()> {
	let keys = ensure_file_array(config, &["llm", "policies", "apiKey", "keys"])?;
	let lookup_id = previous_id.unwrap_or(&prepared.id);
	if let Some(existing) = keys
		.iter()
		.enumerate()
		.position(|(index, value)| file_api_key_id(value, index) == lookup_id)
	{
		keys[existing] = prepared.value.clone();
	} else if previous_id.is_some() {
		return Err(
			ConfigResourceError::NotFound(format!(
				"config resource not found: {}/{}",
				prepared.kind, lookup_id
			))
			.into(),
		);
	} else {
		keys.push(prepared.value.clone());
	}
	Ok(())
}

fn delete_file_api_key(config: &mut Value, id: &str) -> anyhow::Result<bool> {
	let Some(keys) = crate::json::traverse_mut(config, &["llm", "policies", "apiKey", "keys"])
		.and_then(Value::as_array_mut)
	else {
		return Ok(false);
	};
	if let Some(index) = keys
		.iter()
		.enumerate()
		.position(|(index, value)| file_api_key_id(value, index) == id)
	{
		keys.remove(index);
		return Ok(true);
	}
	Ok(false)
}

/// Projects the singleton surface settings resource onto its top-level fields.
fn upsert_file_settings(
	config: &mut Value,
	kind: ConfigResourceKind,
	value: &Value,
) -> anyhow::Result<()> {
	let (section, fields) = kind.settings_fields().expect("settings resource");
	let value = value
		.as_object()
		.ok_or_else(|| anyhow::anyhow!("{kind}/default must be an object"))?;
	let settings = ensure_file_object(config, &[section])?;
	for &field in fields {
		if let Some(value) = value.get(field) {
			settings.insert(field.to_string(), value.clone());
		} else {
			settings.remove(field);
		}
	}
	Ok(())
}

fn delete_file_settings(config: &mut Value, kind: ConfigResourceKind) -> anyhow::Result<bool> {
	let (section, fields) = kind.settings_fields().expect("settings resource");
	let Some(settings) = crate::json::traverse_mut(config, &[section]).and_then(Value::as_object_mut)
	else {
		return Ok(false);
	};
	let mut deleted = false;
	for &field in fields {
		deleted |= settings.remove(field).is_some();
	}
	Ok(deleted)
}

/// Replaces only the inline catalog overlay, preserving file-backed catalog sources.
fn upsert_file_model_catalog(config: &mut Value, value: &Value) -> anyhow::Result<()> {
	let value = value
		.as_object()
		.ok_or_else(|| anyhow::anyhow!("modelCatalog resource must be an object"))?;
	let sources = ensure_file_array(config, &["config", "modelCatalog"])?;
	sources.retain(|source| source.get("inline").is_none());
	if let Some(custom) = value.get("custom") {
		sources.push(serde_json::json!({ "inline": custom }));
	}
	Ok(())
}

fn delete_file_model_catalog(config: &mut Value) -> anyhow::Result<bool> {
	let Some(sources) =
		crate::json::traverse_mut(config, &["config", "modelCatalog"]).and_then(Value::as_array_mut)
	else {
		return Ok(false);
	};
	let before = sources.len();
	sources.retain(|source| source.get("inline").is_none());
	Ok(sources.len() != before)
}

pub(crate) fn prepare_file_api_key_update(
	id: String,
	mut value: Value,
	created_at: Option<i64>,
) -> anyhow::Result<PreparedResource> {
	validate_id(&id)?;
	validate_api_key_metadata(&value)?;
	if !id.starts_with("@index:") {
		set_api_key_managed_metadata(&mut value, id.clone(), created_at)?;
	}
	Ok(PreparedResource::new(
		ConfigResourceKind::LlmApiKey,
		id,
		value,
	))
}
pub(crate) fn prepare_resources(
	kind: ConfigResourceKind,
	request: ConfigResourceUpsertRequest,
) -> anyhow::Result<Vec<PreparedResource>> {
	request
		.resources
		.into_iter()
		.map(|resource| prepare_resource(kind, resource.value))
		.collect()
}

pub(crate) fn prepare_api_key_update(
	id: String,
	mut value: Value,
	created_at: Option<i64>,
) -> anyhow::Result<PreparedResource> {
	validate_id(&id)?;
	validate_api_key_metadata(&value)?;
	set_api_key_managed_metadata(&mut value, id.clone(), created_at)?;
	Ok(PreparedResource::new(
		ConfigResourceKind::LlmApiKey,
		id,
		value,
	))
}

pub(crate) fn prepare_policy_upsert(
	kind: ConfigResourceKind,
	id: String,
	value: Value,
) -> anyhow::Result<PreparedResource> {
	validate_id(&id)?;
	if !matches!(
		kind,
		ConfigResourceKind::LlmPolicy
			| ConfigResourceKind::McpPolicy
			| ConfigResourceKind::UiPolicy
			| ConfigResourceKind::FrontendPolicy
	) {
		return Err(
			ConfigResourceError::InvalidRequest(format!("{kind} is not a policy resource")).into(),
		);
	}
	if kind == ConfigResourceKind::LlmPolicy && id == "apiKey" && value.get("keys").is_some() {
		return Err(
			ConfigResourceError::InvalidRequest(
				"llm.policy/apiKey must not include keys; use llm.apiKey resources".to_string(),
			)
			.into(),
		);
	}
	Ok(PreparedResource::new(kind, id, value))
}

pub fn merge_model_catalog_sources(
	resources: &[ConfigResource],
	mut configured: Vec<crate::ModelCatalogSource>,
) -> anyhow::Result<Vec<crate::ModelCatalogSource>> {
	let Some(resource) = resources
		.iter()
		.find(|resource| resource.kind == ConfigResourceKind::ModelCatalog)
	else {
		return Ok(configured);
	};
	let value = resource
		.value
		.as_object()
		.ok_or_else(|| anyhow::anyhow!("modelCatalog resource must be an object"))?;
	let mut sources = Vec::new();
	if let Some(base) = value.get("base") {
		let mut inline: crate::llm::catalog::Catalog = serde_json::from_value(base.clone())?;
		if inline.metadata.is_none() {
			inline.metadata = Some(crate::llm::catalog::CatalogMetadata {
				source: None,
				// Legacy base catalogs predate generatedAt. Treat them as older than every
				// timestamped catalog rather than guessing from the resource timestamp,
				// which may also reflect an unrelated custom-overlay edit.
				generated_at: DateTime::<Utc>::UNIX_EPOCH,
				unknown: Default::default(),
			});
		}
		sources.push(crate::ModelCatalogSource::InlineCatalog { inline });
	}
	if let Some(custom) = value.get("custom") {
		sources.push(crate::ModelCatalogSource::InlineCatalog {
			inline: serde_json::from_value(custom.clone())?,
		});
	}
	sources.append(&mut configured);
	Ok(sources)
}

pub(crate) fn apply_prepared_upsert(
	mut resources: Vec<ConfigResource>,
	prepared: &[PreparedResource],
) -> anyhow::Result<Vec<ConfigResource>> {
	for prepared in prepared {
		validate_id(&prepared.id)?;
		if let Some(existing) = resources.iter_mut().find(|resource| {
			resource.kind == prepared.kind
				&& resource.id == prepared.id
				&& resource.scopes == prepared.target_scopes()
		}) {
			existing.value = prepared.value.clone();
			existing.scopes = prepared.scopes.clone();
			continue;
		}
		resources.push(ConfigResource {
			kind: prepared.kind,
			id: prepared.id.clone(),
			scopes: prepared.scopes.clone(),
			value: prepared.value.clone(),
			revision: 1,
			created_at: Utc::now(),
			updated_at: Utc::now(),
		});
	}
	Ok(resources)
}

/// Removes the global row `(kind, id)` from an in-memory resource list.
pub(crate) fn apply_delete(
	resources: Vec<ConfigResource>,
	kind: ConfigResourceKind,
	id: &str,
) -> Vec<ConfigResource> {
	apply_delete_scoped(resources, kind, id, &global_scopes())
}

/// Removes the row `(kind, id, scopes)` from an in-memory resource list.
pub(crate) fn apply_delete_scoped(
	resources: Vec<ConfigResource>,
	kind: ConfigResourceKind,
	id: &str,
	scopes: &[String],
) -> Vec<ConfigResource> {
	resources
		.into_iter()
		.filter(|resource| !(resource.kind == kind && resource.id == id && resource.scopes == scopes))
		.collect()
}

pub async fn materialize_hybrid_config(
	source: &crate::ConfigSource,
	store: &ConfigResourceStore,
) -> anyhow::Result<String> {
	let base = source.read_to_string().await?;
	let resources = store.list(None).await?;
	materialize_config(base.as_str(), &resources)
}

pub(crate) fn materialize_config(
	base: &str,
	resources: &[ConfigResource],
) -> anyhow::Result<String> {
	let mut config: Value = crate::yaml::from_str(base)?;
	overlay_config_resources(&mut config, resources)?;
	crate::yaml::to_string(&config)
}

fn overlay_config_resources(
	config: &mut Value,
	resources: &[ConfigResource],
) -> anyhow::Result<()> {
	let has_llm_resources = resources.iter().any(|resource| {
		matches!(
			resource.kind,
			ConfigResourceKind::LlmSettings
				| ConfigResourceKind::LlmProvider
				| ConfigResourceKind::LlmModel
				| ConfigResourceKind::LlmVirtualModel
				| ConfigResourceKind::LlmApiKey
				| ConfigResourceKind::LlmPolicy
		)
	});
	let has_ui_resources = resources
		.iter()
		.any(|resource| resource.kind == ConfigResourceKind::UiPolicy);
	let has_frontend_resources = resources
		.iter()
		.any(|resource| resource.kind == ConfigResourceKind::FrontendPolicy);
	let has_model_catalog = resources
		.iter()
		.any(|resource| resource.kind == ConfigResourceKind::ModelCatalog);
	let has_mcp_resources = resources.iter().any(|resource| {
		matches!(
			resource.kind,
			ConfigResourceKind::McpTarget
				| ConfigResourceKind::McpPolicy
				| ConfigResourceKind::McpSettings
		)
	});
	let has_traffic_resources = resources.iter().any(|resource| {
		matches!(
			resource.kind,
			ConfigResourceKind::TrafficGateway
				| ConfigResourceKind::TrafficRoute
				| ConfigResourceKind::TrafficTcpRoute
		)
	});
	let has_llm_policies = resources
		.iter()
		.any(|resource| resource.kind == ConfigResourceKind::LlmPolicy);
	if !has_llm_resources
		&& !has_mcp_resources
		&& !has_traffic_resources
		&& !has_ui_resources
		&& !has_frontend_resources
		&& !has_model_catalog
	{
		return Ok(());
	}

	if has_model_catalog {
		let configured = config
			.pointer("/config/modelCatalog")
			.cloned()
			.map(serde_json::from_value)
			.transpose()?
			.unwrap_or_default();
		let sources = merge_model_catalog_sources(resources, configured)?;
		*ensure_file_array(config, &["config", "modelCatalog"])? = serde_json::to_value(sources)?
			.as_array()
			.expect("model catalog sources serialize as an array")
			.clone();
	}

	let Some(root) = config.as_object_mut() else {
		anyhow::bail!("local config root must be a JSON object");
	};
	if has_llm_resources {
		if has_llm_policies
			&& !root.contains_key("llm")
			&& !resources
				.iter()
				.any(|r| r.kind == ConfigResourceKind::LlmSettings)
		{
			return Err(
				ConfigResourceError::Conflict(
					"DB-backed LLM policies require llm in the file config or a llm.settings resource"
						.to_string(),
				)
				.into(),
			);
		}
		let llm = root
			.entry("llm")
			.or_insert_with(|| Value::Object(serde_json::Map::new()));
		let Some(llm) = llm.as_object_mut() else {
			if has_llm_policies {
				return Err(
					ConfigResourceError::Conflict(
						"DB-backed LLM policies require llm to be an object in the file config".to_string(),
					)
					.into(),
				);
			}
			anyhow::bail!("local config llm must be a JSON object");
		};

		append_settings(llm, resources, ConfigResourceKind::LlmSettings)?;
		append_policy_kind(
			llm,
			"policies",
			resources,
			ConfigResourceKind::LlmPolicy,
			"llm.policies",
		)?;
		append_llm_kind(llm, resources, ConfigResourceKind::LlmProvider, "providers")?;
		append_llm_kind(llm, resources, ConfigResourceKind::LlmModel, "models")?;
		append_llm_kind(
			llm,
			resources,
			ConfigResourceKind::LlmVirtualModel,
			"virtualModels",
		)?;
		append_api_keys(llm, resources)?;
	}
	if has_mcp_resources {
		let mcp = root
			.entry("mcp")
			.or_insert_with(|| Value::Object(serde_json::Map::new()));
		let Some(mcp) = mcp.as_object_mut() else {
			anyhow::bail!("local config mcp must be a JSON object");
		};
		mcp
			.entry("targets")
			.or_insert_with(|| Value::Array(Vec::new()));

		append_settings(mcp, resources, ConfigResourceKind::McpSettings)?;
		append_policy_kind(
			mcp,
			"policies",
			resources,
			ConfigResourceKind::McpPolicy,
			"mcp.policies",
		)?;
		append_list_kind(
			mcp,
			resources,
			ConfigResourceKind::McpTarget,
			"targets",
			"mcp",
		)?;
	}
	if has_traffic_resources {
		append_traffic_gateways(root, resources)?;
		append_traffic_routes(root, resources, ConfigResourceKind::TrafficRoute, "routes")?;
		append_traffic_routes(
			root,
			resources,
			ConfigResourceKind::TrafficTcpRoute,
			"tcpRoutes",
		)?;
	}
	if has_ui_resources {
		let Some(ui) = root.get_mut("ui") else {
			return Err(
				ConfigResourceError::Conflict(
					"DB-backed UI policies require ui in the file config".to_string(),
				)
				.into(),
			);
		};
		let Some(ui) = ui.as_object_mut() else {
			return Err(
				ConfigResourceError::Conflict(
					"DB-backed UI policies require ui to be an object in the file config".to_string(),
				)
				.into(),
			);
		};
		append_policy_kind(
			ui,
			"policies",
			resources,
			ConfigResourceKind::UiPolicy,
			"ui.policies",
		)?;
	}
	if has_frontend_resources {
		append_policy_kind(
			root,
			"frontendPolicies",
			resources,
			ConfigResourceKind::FrontendPolicy,
			"frontendPolicies",
		)?;
	}
	Ok(())
}

fn append_policy_kind(
	section: &mut serde_json::Map<String, Value>,
	key: &str,
	resources: &[ConfigResource],
	kind: ConfigResourceKind,
	path: &str,
) -> anyhow::Result<()> {
	let Some(db_resources) = non_empty_resources(resources, kind) else {
		return Ok(());
	};
	let policies = section
		.entry(key)
		.or_insert_with(|| Value::Object(serde_json::Map::new()));
	if policies.is_null() {
		*policies = Value::Object(serde_json::Map::new());
	}
	let policies = policies.as_object_mut().ok_or_else(|| {
		ConfigResourceError::Conflict(format!(
			"DB-backed {kind} resources require {path} to be an object in the file config"
		))
	})?;
	for resource in db_resources {
		if policies.contains_key(&resource.id) {
			return Err(
				ConfigResourceError::Conflict(format!(
					"config resource {kind}/{} conflicts with file-owned resource",
					resource.id
				))
				.into(),
			);
		}
		let mut value = resource.value.clone();
		if kind == ConfigResourceKind::LlmPolicy && resource.id == "apiKey" {
			let Some(policy) = value.as_object_mut() else {
				return Err(
					ConfigResourceError::InvalidRequest("llm.policy/apiKey must be an object".to_string())
						.into(),
				);
			};
			if policy.contains_key("keys") {
				return Err(
					ConfigResourceError::InvalidRequest(
						"llm.policy/apiKey must not include keys; use llm.apiKey resources".to_string(),
					)
					.into(),
				);
			}
			policy.insert("keys".to_string(), Value::Array(Vec::new()));
		}
		policies.insert(resource.id.clone(), value);
	}
	Ok(())
}

fn append_api_keys(
	llm: &mut serde_json::Map<String, Value>,
	resources: &[ConfigResource],
) -> anyhow::Result<()> {
	let Some(db_resources) = non_empty_resources(resources, ConfigResourceKind::LlmApiKey) else {
		return Ok(());
	};
	let policies = llm
		.get_mut("policies")
		.ok_or_else(|| {
			ConfigResourceError::Conflict("DB-backed API keys require llm.policies.apiKey".to_string())
		})?
		.as_object_mut()
		.ok_or_else(|| anyhow::anyhow!("local config llm.policies must be an object"))?;
	let policy = policies
		.get_mut("apiKey")
		.ok_or_else(|| {
			ConfigResourceError::Conflict("DB-backed API keys require llm.policies.apiKey".to_string())
		})?
		.as_object_mut()
		.ok_or_else(|| anyhow::anyhow!("local config llm.policies.apiKey must be an object"))?;
	let keys = policy
		.entry("keys")
		.or_insert_with(|| Value::Array(Vec::new()))
		.as_array_mut()
		.ok_or_else(|| anyhow::anyhow!("local config llm.policies.apiKey.keys must be an array"))?;

	for resource in db_resources {
		let mut value = resource.value.clone();
		// Keys from scoped rows get budget counters of their own.
		if !is_global_scopes(&resource.scopes) {
			value["metadata"][API_KEY_SCOPES_METADATA] = Value::from(resource.scopes.clone());
		}
		keys.push(value);
	}
	Ok(())
}

fn append_list_kind(
	section: &mut serde_json::Map<String, Value>,
	resources: &[ConfigResource],
	kind: ConfigResourceKind,
	field: &str,
	section_name: &str,
) -> anyhow::Result<()> {
	let Some(db_resources) = non_empty_resources(resources, kind) else {
		return Ok(());
	};

	let values = section
		.entry(field)
		.or_insert_with(|| Value::Array(Vec::new()));
	let Some(values) = values.as_array_mut() else {
		anyhow::bail!("local config {section_name}.{field} must be an array");
	};

	for existing in values.iter() {
		let id = resource_id(kind, existing)?;
		if db_resources.iter().any(|resource| resource.id == id) {
			return Err(
				ConfigResourceError::Conflict(format!(
					"config resource {kind}/{id} conflicts with file-owned resource"
				))
				.into(),
			);
		}
	}

	for resource in db_resources {
		values.push(resource.value.clone());
	}
	Ok(())
}

fn append_llm_kind(
	llm: &mut serde_json::Map<String, Value>,
	resources: &[ConfigResource],
	kind: ConfigResourceKind,
	field: &str,
) -> anyhow::Result<()> {
	append_list_kind(llm, resources, kind, field, "llm")
}

fn append_settings(
	settings: &mut serde_json::Map<String, Value>,
	resources: &[ConfigResource],
	kind: ConfigResourceKind,
) -> anyhow::Result<()> {
	let (_, fields) = kind.settings_fields().expect("settings resource");
	let Some(resource) = resources.iter().find(|resource| resource.kind == kind) else {
		return Ok(());
	};
	let value = resource.value.as_object().ok_or_else(|| {
		ConfigResourceError::InvalidRequest(format!("{kind}/default must be an object"))
	})?;
	for (field, value) in value {
		if !fields.contains(&field.as_str()) {
			return Err(
				ConfigResourceError::InvalidRequest(format!(
					"{kind}/default contains unsupported field: {field}"
				))
				.into(),
			);
		}
		if settings.contains_key(field) {
			return Err(
				ConfigResourceError::Conflict(format!(
					"config resource {kind}/default field {field} conflicts with file-owned configuration"
				))
				.into(),
			);
		}
		settings.insert(field.clone(), value.clone());
	}
	Ok(())
}

fn append_traffic_gateways(
	root: &mut serde_json::Map<String, Value>,
	resources: &[ConfigResource],
) -> anyhow::Result<()> {
	let Some(db_resources) = non_empty_resources(resources, ConfigResourceKind::TrafficGateway)
	else {
		return Ok(());
	};
	let gateways = root
		.entry("gateways")
		.or_insert_with(|| Value::Object(serde_json::Map::new()));
	let Some(gateways) = gateways.as_object_mut() else {
		anyhow::bail!("local config gateways must be a JSON object");
	};
	for resource in db_resources {
		if gateways.contains_key(&resource.id) {
			return Err(
				ConfigResourceError::Conflict(format!(
					"config resource traffic.gateway/{} conflicts with file-owned resource",
					resource.id
				))
				.into(),
			);
		}
		let mut value = resource.value.clone();
		let Some(value) = value.as_object_mut() else {
			return Err(
				ConfigResourceError::InvalidRequest(format!(
					"traffic.gateway/{} must be an object",
					resource.id
				))
				.into(),
			);
		};
		value.remove("name");
		gateways.insert(resource.id.clone(), Value::Object(std::mem::take(value)));
	}
	Ok(())
}

fn append_traffic_routes(
	root: &mut serde_json::Map<String, Value>,
	resources: &[ConfigResource],
	kind: ConfigResourceKind,
	field: &str,
) -> anyhow::Result<()> {
	let Some(mut db_resources) = non_empty_resources(resources, kind) else {
		return Ok(());
	};
	db_resources.sort_by(|left, right| left.id.cmp(&right.id));
	let routes = root
		.entry(field)
		.or_insert_with(|| Value::Array(Vec::new()));
	let Some(routes) = routes.as_array_mut() else {
		anyhow::bail!("local config {field} must be an array");
	};
	for existing in routes.iter() {
		let Some(name) = existing.get("name").and_then(Value::as_str) else {
			continue;
		};
		if db_resources.iter().any(|resource| resource.id == name) {
			return Err(
				ConfigResourceError::Conflict(format!(
					"config resource {kind}/{name} conflicts with file-owned resource"
				))
				.into(),
			);
		}
	}
	for resource in db_resources {
		routes.push(resource.value.clone());
	}
	Ok(())
}

fn non_empty_resources(
	resources: &[ConfigResource],
	kind: ConfigResourceKind,
) -> Option<Vec<&ConfigResource>> {
	let resources = resources
		.iter()
		.filter(|resource| resource.kind == kind)
		.collect::<Vec<_>>();
	(!resources.is_empty()).then_some(resources)
}

pub(crate) fn prepare_resource(
	kind: ConfigResourceKind,
	mut value: Value,
) -> anyhow::Result<PreparedResource> {
	let id = match kind {
		ConfigResourceKind::LlmApiKey => {
			validate_api_key_metadata(&value)?;
			let id = uuid::Uuid::new_v4().to_string();
			set_api_key_managed_metadata(&mut value, id.clone(), Some(Utc::now().timestamp()))?;
			id
		},
		ConfigResourceKind::LlmPolicy
		| ConfigResourceKind::McpPolicy
		| ConfigResourceKind::UiPolicy
		| ConfigResourceKind::FrontendPolicy => {
			return Err(
				ConfigResourceError::InvalidRequest(format!("{kind} resources require an item ID")).into(),
			);
		},
		_ => resource_id(kind, &value)?,
	};
	Ok(PreparedResource::new(kind, id, value))
}

fn resource_id(kind: ConfigResourceKind, value: &Value) -> anyhow::Result<String> {
	match kind {
		ConfigResourceKind::ModelCatalog => Ok("default".to_string()),
		ConfigResourceKind::McpSettings | ConfigResourceKind::LlmSettings => Ok("default".to_string()),
		ConfigResourceKind::LlmProvider
		| ConfigResourceKind::LlmVirtualModel
		| ConfigResourceKind::McpTarget
		| ConfigResourceKind::TrafficGateway
		| ConfigResourceKind::TrafficRoute
		| ConfigResourceKind::TrafficTcpRoute => string_field(value, "name", || {
			format!("{kind} resources require value.name")
		}),
		ConfigResourceKind::LlmModel => string_field(value, "id", || {
			"llm.model resources require value.id or value.name".to_string()
		})
		.or_else(|_| {
			string_field(value, "name", || {
				"llm.model resources require value.id or value.name".to_string()
			})
		}),
		ConfigResourceKind::LlmApiKey => value
			.get("metadata")
			.and_then(Value::as_object)
			.and_then(|metadata| metadata.get(API_KEY_ID_METADATA))
			.and_then(Value::as_str)
			.map(ToString::to_string)
			.ok_or_else(|| {
				ConfigResourceError::InvalidRequest(format!(
					"llm.apiKey resources require value.metadata.{API_KEY_ID_METADATA}"
				))
				.into()
			}),
		ConfigResourceKind::LlmPolicy
		| ConfigResourceKind::McpPolicy
		| ConfigResourceKind::UiPolicy
		| ConfigResourceKind::FrontendPolicy => Err(
			ConfigResourceError::InvalidRequest(format!("{kind} resources require an item ID")).into(),
		),
	}
}

fn validate_api_key_metadata(value: &Value) -> anyhow::Result<()> {
	if let Some(field) = value
		.get("metadata")
		.and_then(Value::as_object)
		.and_then(|metadata| {
			metadata
				.keys()
				// Key hint is not really required to be trusted so we can allow that
				.find(|field| field.starts_with(API_KEY_METADATA_PREFIX) && *field != API_KEY_HINT_METADATA)
		}) {
		return Err(
			ConfigResourceError::InvalidRequest(format!(
				"llm.apiKey metadata field {field} uses the reserved agentgateway.dev/ prefix"
			))
			.into(),
		);
	}
	Ok(())
}

fn set_api_key_managed_metadata(
	value: &mut Value,
	id: String,
	created_at: Option<i64>,
) -> anyhow::Result<()> {
	let Some(object) = value.as_object_mut() else {
		return Err(
			ConfigResourceError::InvalidRequest("llm.apiKey resources must be JSON objects".to_string())
				.into(),
		);
	};
	let metadata = object
		.entry("metadata")
		.or_insert_with(|| Value::Object(serde_json::Map::new()));
	let Some(metadata) = metadata.as_object_mut() else {
		return Err(
			ConfigResourceError::InvalidRequest(
				"llm.apiKey resources require value.metadata to be an object".to_string(),
			)
			.into(),
		);
	};
	metadata.insert(API_KEY_ID_METADATA.to_string(), Value::String(id));
	if let Some(created_at) = created_at {
		metadata.insert(
			API_KEY_CREATED_AT_METADATA.to_string(),
			Value::Number(created_at.into()),
		);
	}
	Ok(())
}

fn string_field(
	value: &Value,
	field: &str,
	error: impl FnOnce() -> String,
) -> anyhow::Result<String> {
	value
		.get(field)
		.and_then(Value::as_str)
		.filter(|value| !value.is_empty())
		.map(ToString::to_string)
		.ok_or_else(|| ConfigResourceError::InvalidRequest(error()).into())
}

const SELECT_RESOURCES: &str =
	"SELECT kind, id, scopes, value_json, revision, created_at, updated_at FROM agw_config_resources";

fn sqlite_scopes(scopes: &[String]) -> String {
	serde_json::to_string(scopes).expect("strings serialize")
}

/// Runs the write hook, if any, inside a Postgres write transaction.
async fn run_write_hook(
	hook: Option<&dyn ConfigWriteHook>,
	tx: &mut Transaction<'_, Postgres>,
	kind: ConfigResourceKind,
	id: &str,
	previous_scopes: Option<&[String]>,
	scopes: Option<&[String]>,
) -> anyhow::Result<()> {
	let Some(hook) = hook else {
		return Ok(());
	};
	let write = ConfigWrite {
		kind,
		id: id.to_string(),
		previous_scopes: previous_scopes.map(<[String]>::to_vec),
		scopes: scopes.map(<[String]>::to_vec),
	};
	hook.before_write(tx, &write).await
}

async fn list_sqlite(
	pool: &SqlitePool,
	kind: Option<ConfigResourceKind>,
	scopes: Option<&[String]>,
) -> anyhow::Result<Vec<ConfigResource>> {
	let mut qb = QueryBuilder::<Sqlite>::new(SELECT_RESOURCES);
	qb.push(" WHERE deleted_at IS NULL");
	if let Some(kind) = kind {
		qb.push(" AND kind = ").push_bind(kind.as_str());
	}
	if let Some(scopes) = scopes {
		qb.push(
			" AND EXISTS (SELECT 1 FROM json_each(scopes) WHERE value IN (SELECT value FROM json_each(",
		)
		.push_bind(sqlite_scopes(scopes))
		.push(")))");
	}
	qb.push(" ORDER BY kind, id, scopes");
	let rows = qb.build().fetch_all(pool).await?;
	rows.into_iter().map(sqlite_row_to_resource).collect()
}

async fn list_postgres(
	pool: &PgPool,
	kind: Option<ConfigResourceKind>,
	scopes: Option<&[String]>,
) -> anyhow::Result<Vec<ConfigResource>> {
	let mut qb = QueryBuilder::<Postgres>::new(SELECT_RESOURCES);
	qb.push(" WHERE deleted_at IS NULL");
	if let Some(kind) = kind {
		qb.push(" AND kind = ").push_bind(kind.as_str());
	}
	if let Some(scopes) = scopes {
		qb.push(" AND scopes && ").push_bind(scopes.to_vec());
	}
	qb.push(" ORDER BY kind, id, scopes");
	let rows = qb.build().fetch_all(pool).await?;
	rows.into_iter().map(postgres_row_to_resource).collect()
}

async fn upsert_sqlite(
	pool: &SqlitePool,
	prepared: Vec<PreparedResource>,
) -> anyhow::Result<(Vec<ConfigResource>, Vec<String>)> {
	let mut tx = pool.begin().await?;
	let mut changed = Vec::with_capacity(prepared.len());
	let mut changed_scopes = Vec::new();
	for resource in prepared {
		let target = resource.target_scopes().to_vec();
		let now = Utc::now().to_rfc3339();
		if target != resource.scopes {
			// Move the row to its new scopes in place, replacing a deleted row holding that key.
			sqlx::query(
				"DELETE FROM agw_config_resources \
				 WHERE kind = ? AND id = ? AND scopes = ? AND deleted_at IS NOT NULL",
			)
			.bind(resource.kind.as_str())
			.bind(&resource.id)
			.bind(sqlite_scopes(&resource.scopes))
			.execute(&mut *tx)
			.await?;
			let moved = sqlx::query(
				"UPDATE agw_config_resources SET scopes = ?, value_json = ?, \
				 revision = revision + 1, updated_at = ? \
				 WHERE kind = ? AND id = ? AND scopes = ? AND deleted_at IS NULL",
			)
			.bind(sqlite_scopes(&resource.scopes))
			.bind(serde_json::to_string(&resource.value)?)
			.bind(&now)
			.bind(resource.kind.as_str())
			.bind(&resource.id)
			.bind(sqlite_scopes(&target))
			.execute(&mut *tx)
			.await?;
			if moved.rows_affected() == 0 {
				return Err(not_found(resource.kind, &resource.id));
			}
		} else {
			sqlx::query(
				"INSERT INTO agw_config_resources \
				 (kind, id, scopes, value_json, revision, created_at, updated_at, deleted_at) \
				 VALUES (?, ?, ?, ?, 1, ?, ?, NULL) \
				 ON CONFLICT(kind, id, scopes) DO UPDATE SET \
					value_json = excluded.value_json, \
					revision = agw_config_resources.revision + 1, \
					updated_at = excluded.updated_at, \
					deleted_at = NULL",
			)
			.bind(resource.kind.as_str())
			.bind(&resource.id)
			.bind(sqlite_scopes(&resource.scopes))
			.bind(serde_json::to_string(&resource.value)?)
			.bind(&now)
			.bind(&now)
			.execute(&mut *tx)
			.await?;
		}
		changed_scopes.extend(target);
		changed_scopes.extend(resource.scopes.iter().cloned());
		changed.push(resource);
	}

	let mut resources = Vec::with_capacity(changed.len());
	for resource in changed {
		if let Some(resource) =
			fetch_sqlite_resource(&mut tx, resource.kind, &resource.id, &resource.scopes).await?
		{
			resources.push(resource);
		}
	}
	tx.commit().await?;
	Ok((resources, canonical_scopes(changed_scopes)))
}

async fn upsert_postgres(
	pool: &PgPool,
	prepared: Vec<PreparedResource>,
	notification_id: &str,
	hook: Option<&dyn ConfigWriteHook>,
) -> anyhow::Result<(Vec<ConfigResource>, Vec<String>)> {
	let mut tx = pool.begin().await?;
	let mut changed = Vec::with_capacity(prepared.len());
	let mut changed_scopes = Vec::new();
	for resource in prepared {
		let target = resource.target_scopes().to_vec();
		let now = Utc::now();
		run_write_hook(
			hook,
			&mut tx,
			resource.kind,
			&resource.id,
			Some(&target),
			Some(&resource.scopes),
		)
		.await?;
		if target != resource.scopes {
			// Move the row to its new scopes in place, replacing a deleted row holding that key.
			sqlx::query(
				"DELETE FROM agw_config_resources \
				 WHERE kind = $1 AND id = $2 AND scopes = $3 AND deleted_at IS NOT NULL",
			)
			.bind(resource.kind.as_str())
			.bind(&resource.id)
			.bind(&resource.scopes)
			.execute(&mut *tx)
			.await?;
			let moved = sqlx::query(
				"UPDATE agw_config_resources SET scopes = $1, value_json = $2, \
				 revision = revision + 1, updated_at = $3 \
				 WHERE kind = $4 AND id = $5 AND scopes = $6 AND deleted_at IS NULL",
			)
			.bind(&resource.scopes)
			.bind(Json(&resource.value))
			.bind(now)
			.bind(resource.kind.as_str())
			.bind(&resource.id)
			.bind(&target)
			.execute(&mut *tx)
			.await?;
			if moved.rows_affected() == 0 {
				return Err(not_found(resource.kind, &resource.id));
			}
		} else {
			sqlx::query(
				"INSERT INTO agw_config_resources \
				 (kind, id, scopes, value_json, revision, created_at, updated_at, deleted_at) \
				 VALUES ($1, $2, $3, $4, 1, $5, $5, NULL) \
				 ON CONFLICT(kind, id, scopes) DO UPDATE SET \
					value_json = excluded.value_json, \
					revision = agw_config_resources.revision + 1, \
					updated_at = excluded.updated_at, \
					deleted_at = NULL",
			)
			.bind(resource.kind.as_str())
			.bind(&resource.id)
			.bind(&resource.scopes)
			.bind(Json(&resource.value))
			.bind(now)
			.execute(&mut *tx)
			.await?;
		}
		changed_scopes.extend(target);
		changed_scopes.extend(resource.scopes.iter().cloned());
		changed.push(resource);
	}

	let mut resources = Vec::with_capacity(changed.len());
	for resource in changed {
		if let Some(resource) =
			fetch_postgres_resource(&mut tx, resource.kind, &resource.id, &resource.scopes).await?
		{
			resources.push(resource);
		}
	}
	let changed_scopes = canonical_scopes(changed_scopes);
	if !resources.is_empty() {
		notify_postgres(&mut tx, notification_id, &changed_scopes).await?;
	}
	tx.commit().await?;
	Ok((resources, changed_scopes))
}

fn not_found(kind: ConfigResourceKind, id: &str) -> anyhow::Error {
	ConfigResourceError::NotFound(format!("config resource not found: {kind}/{id}")).into()
}

async fn rename_sqlite(
	pool: &SqlitePool,
	previous_kind: ConfigResourceKind,
	previous_id: &str,
	prepared: PreparedResource,
) -> anyhow::Result<ConfigResource> {
	if previous_kind != prepared.kind || previous_id == prepared.id {
		return Err(
			ConfigResourceError::InvalidRequest("config resource rename requires a new ID".to_string())
				.into(),
		);
	}
	let target = prepared.target_scopes().to_vec();
	let mut tx = pool.begin().await?;
	let now = Utc::now().to_rfc3339();
	if !soft_delete_sqlite(&mut tx, previous_kind, previous_id, &target, &now).await? {
		return Err(
			ConfigResourceError::NotFound(format!(
				"config resource not found: {previous_kind}/{previous_id}"
			))
			.into(),
		);
	}

	let inserted = sqlx::query(
		"INSERT INTO agw_config_resources \
		 (kind, id, scopes, value_json, revision, created_at, updated_at, deleted_at) \
		 VALUES (?, ?, ?, ?, 1, ?, ?, NULL) \
		 ON CONFLICT(kind, id, scopes) DO UPDATE SET \
			value_json = excluded.value_json, \
			revision = agw_config_resources.revision + 1, \
			updated_at = excluded.updated_at, \
			deleted_at = NULL \
		 WHERE agw_config_resources.deleted_at IS NOT NULL",
	)
	.bind(prepared.kind.as_str())
	.bind(&prepared.id)
	.bind(sqlite_scopes(&prepared.scopes))
	.bind(serde_json::to_string(&prepared.value)?)
	.bind(&now)
	.bind(&now)
	.execute(&mut *tx)
	.await?;
	if inserted.rows_affected() == 0 {
		return Err(
			ConfigResourceError::Conflict(format!(
				"config resource already exists: {}/{}",
				prepared.kind, prepared.id
			))
			.into(),
		);
	}
	let resource = fetch_sqlite_resource(&mut tx, prepared.kind, &prepared.id, &prepared.scopes)
		.await?
		.ok_or_else(|| anyhow::anyhow!("renamed config resource was not found"))?;
	tx.commit().await?;
	Ok(resource)
}

async fn rename_postgres(
	pool: &PgPool,
	previous_kind: ConfigResourceKind,
	previous_id: &str,
	prepared: PreparedResource,
	notification_id: &str,
	hook: Option<&dyn ConfigWriteHook>,
) -> anyhow::Result<ConfigResource> {
	if previous_kind != prepared.kind || previous_id == prepared.id {
		return Err(
			ConfigResourceError::InvalidRequest("config resource rename requires a new ID".to_string())
				.into(),
		);
	}
	let target = prepared.target_scopes().to_vec();
	let mut tx = pool.begin().await?;
	let now = Utc::now();
	run_write_hook(
		hook,
		&mut tx,
		previous_kind,
		previous_id,
		Some(&target),
		None,
	)
	.await?;
	run_write_hook(
		hook,
		&mut tx,
		prepared.kind,
		&prepared.id,
		None,
		Some(&prepared.scopes),
	)
	.await?;
	if !soft_delete_postgres(&mut tx, previous_kind, previous_id, &target, now).await? {
		return Err(
			ConfigResourceError::NotFound(format!(
				"config resource not found: {previous_kind}/{previous_id}"
			))
			.into(),
		);
	}

	let inserted = sqlx::query(
		"INSERT INTO agw_config_resources \
		 (kind, id, scopes, value_json, revision, created_at, updated_at, deleted_at) \
		 VALUES ($1, $2, $3, $4, 1, $5, $5, NULL) \
		 ON CONFLICT(kind, id, scopes) DO UPDATE SET \
			value_json = excluded.value_json, \
			revision = agw_config_resources.revision + 1, \
			updated_at = excluded.updated_at, \
			deleted_at = NULL \
		 WHERE agw_config_resources.deleted_at IS NOT NULL",
	)
	.bind(prepared.kind.as_str())
	.bind(&prepared.id)
	.bind(&prepared.scopes)
	.bind(Json(&prepared.value))
	.bind(now)
	.execute(&mut *tx)
	.await?;
	if inserted.rows_affected() == 0 {
		return Err(
			ConfigResourceError::Conflict(format!(
				"config resource already exists: {}/{}",
				prepared.kind, prepared.id
			))
			.into(),
		);
	}
	let resource = fetch_postgres_resource(&mut tx, prepared.kind, &prepared.id, &prepared.scopes)
		.await?
		.ok_or_else(|| anyhow::anyhow!("renamed config resource was not found"))?;
	notify_postgres(
		&mut tx,
		notification_id,
		&canonical_scopes([target, prepared.scopes].concat()),
	)
	.await?;
	tx.commit().await?;
	Ok(resource)
}

async fn delete_sqlite(
	pool: &SqlitePool,
	kind: ConfigResourceKind,
	id: &str,
	scopes: &[String],
) -> anyhow::Result<bool> {
	let mut tx = pool.begin().await?;
	let now = Utc::now().to_rfc3339();
	let deleted = soft_delete_sqlite(&mut tx, kind, id, scopes, &now).await?;
	tx.commit().await?;
	Ok(deleted)
}

async fn soft_delete_sqlite(
	tx: &mut Transaction<'_, Sqlite>,
	kind: ConfigResourceKind,
	id: &str,
	scopes: &[String],
	now: &str,
) -> anyhow::Result<bool> {
	let result = sqlx::query(
		"UPDATE agw_config_resources \
		 SET revision = revision + 1, updated_at = ?, deleted_at = ? \
		 WHERE kind = ? AND id = ? AND scopes = ? AND deleted_at IS NULL",
	)
	.bind(now)
	.bind(now)
	.bind(kind.as_str())
	.bind(id)
	.bind(sqlite_scopes(scopes))
	.execute(&mut **tx)
	.await?;
	Ok(result.rows_affected() > 0)
}

async fn delete_postgres(
	pool: &PgPool,
	kind: ConfigResourceKind,
	id: &str,
	scopes: &[String],
	notification_id: &str,
	hook: Option<&dyn ConfigWriteHook>,
) -> anyhow::Result<bool> {
	let mut tx = pool.begin().await?;
	let now = Utc::now();
	run_write_hook(hook, &mut tx, kind, id, Some(scopes), None).await?;
	let deleted = soft_delete_postgres(&mut tx, kind, id, scopes, now).await?;
	if deleted {
		notify_postgres(&mut tx, notification_id, scopes).await?;
	}
	tx.commit().await?;
	Ok(deleted)
}

async fn soft_delete_postgres(
	tx: &mut Transaction<'_, Postgres>,
	kind: ConfigResourceKind,
	id: &str,
	scopes: &[String],
	now: DateTime<Utc>,
) -> anyhow::Result<bool> {
	let result = sqlx::query(
		"UPDATE agw_config_resources \
		 SET revision = revision + 1, updated_at = $1, deleted_at = $1 \
		 WHERE kind = $2 AND id = $3 AND scopes = $4 AND deleted_at IS NULL",
	)
	.bind(now)
	.bind(kind.as_str())
	.bind(id)
	.bind(scopes)
	.execute(&mut **tx)
	.await?;
	Ok(result.rows_affected() > 0)
}

async fn notify_postgres(
	tx: &mut Transaction<'_, Postgres>,
	notification_id: &str,
	scopes: &[String],
) -> anyhow::Result<()> {
	let payload = serde_json::json!({ "origin": notification_id, "scopes": scopes });
	sqlx::query("SELECT pg_notify($1, $2)")
		.bind(POSTGRES_CHANGE_CHANNEL)
		.bind(payload.to_string())
		.execute(&mut **tx)
		.await?;
	Ok(())
}

async fn fetch_sqlite_resource(
	tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
	kind: ConfigResourceKind,
	id: &str,
	scopes: &[String],
) -> anyhow::Result<Option<ConfigResource>> {
	sqlx::query(
		"SELECT kind, id, scopes, value_json, revision, created_at, updated_at \
		 FROM agw_config_resources \
		 WHERE kind = ? AND id = ? AND scopes = ? AND deleted_at IS NULL",
	)
	.bind(kind.as_str())
	.bind(id)
	.bind(sqlite_scopes(scopes))
	.fetch_optional(&mut **tx)
	.await?
	.map(sqlite_row_to_resource)
	.transpose()
}

async fn fetch_postgres_resource(
	tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
	kind: ConfigResourceKind,
	id: &str,
	scopes: &[String],
) -> anyhow::Result<Option<ConfigResource>> {
	sqlx::query(
		"SELECT kind, id, scopes, value_json, revision, created_at, updated_at \
		 FROM agw_config_resources \
		 WHERE kind = $1 AND id = $2 AND scopes = $3 AND deleted_at IS NULL",
	)
	.bind(kind.as_str())
	.bind(id)
	.bind(scopes.to_vec())
	.fetch_optional(&mut **tx)
	.await?
	.map(postgres_row_to_resource)
	.transpose()
}

fn sqlite_row_to_resource(row: sqlx::sqlite::SqliteRow) -> anyhow::Result<ConfigResource> {
	let value_json: String = row.try_get("value_json")?;
	let scopes: String = row.try_get("scopes")?;
	let kind: String = row.try_get("kind")?;
	let created_at: String = row.try_get("created_at")?;
	let updated_at: String = row.try_get("updated_at")?;
	Ok(ConfigResource {
		kind: kind.parse()?,
		id: row.try_get("id")?,
		scopes: serde_json::from_str(&scopes)?,
		value: serde_json::from_str(&value_json)?,
		revision: row.try_get("revision")?,
		created_at: created_at.parse()?,
		updated_at: updated_at.parse()?,
	})
}

fn postgres_row_to_resource(row: sqlx::postgres::PgRow) -> anyhow::Result<ConfigResource> {
	let kind: String = row.try_get("kind")?;
	let Json(value) = row.try_get("value_json")?;
	Ok(ConfigResource {
		kind: kind.parse()?,
		id: row.try_get("id")?,
		scopes: row.try_get("scopes")?,
		value,
		revision: row.try_get("revision")?,
		created_at: row.try_get("created_at")?,
		updated_at: row.try_get("updated_at")?,
	})
}

const POSTGRES_CHANGE_CHANNEL: &str = "agentgateway_config_changed";

#[cfg(test)]
mod tests {
	use serde_json::json;

	use super::*;

	fn test_resource(kind: ConfigResourceKind, id: &str, value: Value) -> ConfigResource {
		ConfigResource {
			kind,
			id: id.to_string(),
			scopes: global_scopes(),
			value,
			revision: 1,
			created_at: Utc::now(),
			updated_at: Utc::now(),
		}
	}

	#[test]
	fn legacy_catalog_base_is_older_than_timestamped_bases() {
		let resource = ConfigResource {
			..test_resource(
				ConfigResourceKind::ModelCatalog,
				"default",
				json!({"base": {"providers": {}}}),
			)
		};

		let sources = merge_model_catalog_sources(&[resource], Vec::new()).unwrap();
		let crate::ModelCatalogSource::InlineCatalog { inline } = &sources[0] else {
			panic!("base must be an inline catalog")
		};
		assert_eq!(
			inline.metadata,
			Some(crate::llm::catalog::CatalogMetadata {
				source: None,
				generated_at: DateTime::<Utc>::UNIX_EPOCH,
				unknown: Default::default(),
			})
		);
	}

	#[test]
	fn derives_resource_ids_and_manages_api_key_ids() {
		let err = "traffic.listener"
			.parse::<ConfigResourceKind>()
			.expect_err("unsupported kind should fail");
		assert!(err.to_string().contains("unsupported config resource kind"));
		assert_eq!(
			resource_id(ConfigResourceKind::LlmProvider, &json!({"name": "openai"}))
				.expect("provider id"),
			"openai"
		);
		assert_eq!(
			resource_id(ConfigResourceKind::LlmModel, &json!({"name": "fast"})).expect("model id"),
			"fast"
		);
		assert_eq!(
			resource_id(
				ConfigResourceKind::LlmModel,
				&json!({"id": "model_01", "name": "fast"})
			)
			.expect("stable model id"),
			"model_01"
		);
		assert_eq!(
			resource_id(
				ConfigResourceKind::LlmVirtualModel,
				&json!({"name": "router"})
			)
			.expect("virtual model id"),
			"router"
		);
		assert_eq!(
			resource_id(
				ConfigResourceKind::McpTarget,
				&json!({"name": "everything"})
			)
			.expect("MCP target id"),
			"everything"
		);
		assert_eq!(
			resource_id(ConfigResourceKind::McpSettings, &json!({})).expect("MCP settings id"),
			"default"
		);
		assert_eq!(
			resource_id(ConfigResourceKind::TrafficRoute, &json!({"name": "api"}))
				.expect("traffic route id"),
			"api"
		);
		assert!(
			resource_id(ConfigResourceKind::TrafficRoute, &json!({})).is_err(),
			"DB-backed routes require a name"
		);

		let created = prepare_resource(
			ConfigResourceKind::LlmApiKey,
			json!({"metadata": {"name": "ci"}}),
		)
		.expect("api key id");
		uuid::Uuid::parse_str(&created.id).expect("UUID v4");
		assert_eq!(
			created
				.value
				.get("metadata")
				.and_then(Value::as_object)
				.and_then(|metadata| metadata.get(API_KEY_ID_METADATA)),
			Some(&Value::String(created.id.clone()))
		);
		let created_at = api_key_created_at(&created.value).expect("managed creation timestamp");
		assert!(created_at > 0);
		let updated = prepare_api_key_update(
			created.id.clone(),
			json!({"metadata": {"name": "updated"}}),
			Some(created_at),
		)
		.expect("API key update");
		assert_eq!(api_key_created_at(&updated.value), Some(created_at));
		assert_eq!(
			api_key_metadata(&updated.value).and_then(|metadata| metadata.get(API_KEY_ID_METADATA)),
			Some(&Value::String(created.id))
		);
		let err = prepare_resource(
			ConfigResourceKind::LlmApiKey,
			json!({"metadata": {"agentgateway.dev/owner": "client"}}),
		)
		.expect_err("reserved API key metadata should fail");
		assert!(
			err
				.to_string()
				.contains("reserved agentgateway.dev/ prefix")
		);
		assert!(
			prepare_api_key_update(
				"key-id".to_string(),
				json!({"metadata": {"agentgateway.dev/owner": "client"}}),
				None,
			)
			.is_err()
		);
		assert!(
			prepare_policy_upsert(
				ConfigResourceKind::LlmPolicy,
				"apiKey".to_string(),
				json!({"keys": []}),
			)
			.is_err()
		);
	}

	fn provider(id: &str, scopes: &[&str]) -> PreparedResource {
		let mut resource = PreparedResource::new(
			ConfigResourceKind::LlmProvider,
			id.to_string(),
			json!({"name": id, "provider": "openAI"}),
		);
		resource.scopes = canonical_scopes(scopes.iter().copied());
		resource
	}

	fn ids_and_scopes(resources: &[ConfigResource]) -> Vec<(&str, Vec<&str>)> {
		resources
			.iter()
			.map(|r| (r.id.as_str(), r.scopes.iter().map(String::as_str).collect()))
			.collect()
	}

	#[tokio::test]
	async fn migrates_legacy_rows_to_global_scope() {
		let pool = DatabasePool::connect("sqlite::memory:").await.unwrap();
		let DatabasePool::Sqlite(sqlite) = &pool else {
			unreachable!()
		};
		sqlx::raw_sql(include_str!(
			"config_store/sqlite_migrations/0001_create_config_resources.sql"
		))
		.execute(sqlite)
		.await
		.unwrap();
		sqlx::query(
			"INSERT INTO agw_config_resources (kind, id, value_json, created_at, updated_at) \
			 VALUES ('llm.provider', 'legacy', '{}', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
		)
		.execute(sqlite)
		.await
		.unwrap();
		let store = ConfigResourceStore::from_pool(pool).await.unwrap();
		let rows = store.list(None).await.unwrap();
		assert_eq!(ids_and_scopes(&rows), vec![("legacy", vec!["global"])]);
	}

	#[tokio::test]
	async fn scopes_filter_rows_and_move_in_place() {
		let store = ConfigResourceStore::connect("sqlite::memory:", None)
			.await
			.unwrap();
		store
			.upsert_prepared(vec![
				provider("shared", &[]),
				provider("shared", &["a"]),
				provider("other", &["b"]),
			])
			.await
			.unwrap();
		let visible = store.list(None).await.unwrap();
		assert_eq!(ids_and_scopes(&visible), vec![("shared", vec!["global"])]);
		store.set_visible_scopes(["a", "global"]);
		assert_eq!(store.list(None).await.unwrap().len(), 2);

		let mut moved = provider("other", &["a", "b"]);
		moved.previous_scopes = Some(vec!["b".into()]);
		let moved = store.upsert_prepared(vec![moved]).await.unwrap().resources;
		assert_eq!(moved[0].revision, 2);
		let all = store.list_scoped(None, ScopeFilter::All).await.unwrap();
		assert_eq!(
			ids_and_scopes(&all),
			vec![
				("other", vec!["a", "b"]),
				("shared", vec!["a"]),
				("shared", vec!["global"])
			]
		);
		store
			.delete_scoped(ConfigResourceKind::LlmProvider, "shared", &["a".into()])
			.await
			.unwrap();
		assert_eq!(store.list(None).await.unwrap().len(), 2);
	}

	#[test]
	fn reloads_follow_visible_scopes() {
		let visible = ["global".to_string()];
		let payload =
			|origin: &str, scopes: &str| format!(r#"{{"origin":"{origin}","scopes":{scopes}}}"#);
		assert!(requires_reload(
			&payload("other", r#"["global"]"#),
			"me",
			&visible
		));
		assert!(!requires_reload(
			&payload("other", r#"["a"]"#),
			"me",
			&visible
		));
		assert!(!requires_reload(
			&payload("me", r#"["global"]"#),
			"me",
			&visible
		));
	}

	#[tokio::test]
	async fn renames_resources_atomically() {
		let store = ConfigResourceStore::connect("sqlite::memory:", None)
			.await
			.expect("connect config resource store");
		store
			.upsert_prepared(vec![PreparedResource::new(
				ConfigResourceKind::LlmProvider,
				"old".to_string(),
				json!({"name": "old", "provider": "openAI"}),
			)])
			.await
			.expect("create resource");
		let response = store
			.rename_prepared(
				ConfigResourceKind::LlmProvider,
				"old",
				PreparedResource::new(
					ConfigResourceKind::LlmProvider,
					"new".to_string(),
					json!({"name": "new", "provider": "openAI"}),
				),
			)
			.await
			.expect("rename resource");
		assert_eq!(response.resources[0].id, "new");
		assert_eq!(
			store
				.list(Some(ConfigResourceKind::LlmProvider))
				.await
				.expect("list resources")
				.into_iter()
				.map(|resource| resource.id)
				.collect::<Vec<_>>(),
			vec!["new"]
		);

		store
			.upsert_prepared(vec![PreparedResource::new(
				ConfigResourceKind::LlmProvider,
				"other".to_string(),
				json!({"name": "other", "provider": "openAI"}),
			)])
			.await
			.expect("create rename target");
		let err = store
			.rename_prepared(
				ConfigResourceKind::LlmProvider,
				"new",
				PreparedResource::new(
					ConfigResourceKind::LlmProvider,
					"other".to_string(),
					json!({"name": "other", "provider": "anthropic"}),
				),
			)
			.await
			.expect_err("rename collision should fail");
		assert!(err.to_string().contains("already exists"));
		assert_eq!(
			store
				.list(Some(ConfigResourceKind::LlmProvider))
				.await
				.expect("list resources after failed rename")
				.into_iter()
				.map(|resource| resource.id)
				.collect::<Vec<_>>(),
			vec!["new", "other"]
		);
	}

	#[test]
	fn materializes_config_resources_into_base_config() {
		let base = r#"
config:
  modelCatalog:
  - inline:
      providers:
        file:
          models:
            file-model: {}
llm:
  models:
  - name: file-model
    provider:
      openAI:
        model: gpt-4o-mini
ui:
  gateways: default
mcp:
  targets:
  - name: file-target
    mcp:
      host: http://localhost:3001/mcp
"#;
		let resources = vec![
			test_resource(
				ConfigResourceKind::ModelCatalog,
				"default",
				json!({
					"custom": {
						"providers": {
							"database": {
								"models": {
									"database-model": {}
								}
							}
						}
					}
				}),
			),
			test_resource(
				ConfigResourceKind::LlmProvider,
				"openai",
				json!({"name": "openai", "provider": "openAI", "params": {"model": "gpt-4o"}}),
			),
			test_resource(
				ConfigResourceKind::LlmModel,
				"model_01",
				json!({"id": "model_01", "name": "db-model", "provider": {"reference": "openai"}}),
			),
			test_resource(
				ConfigResourceKind::LlmVirtualModel,
				"router",
				json!({"name": "router", "routing": {"weighted": {"targets": [{"model": "db-model"}]}}}),
			),
			test_resource(
				ConfigResourceKind::LlmPolicy,
				"cors",
				json!({"allowOrigins": ["https://example.com"]}),
			),
			test_resource(
				ConfigResourceKind::LlmPolicy,
				"apiKey",
				json!({"mode": "strict"}),
			),
			test_resource(
				ConfigResourceKind::LlmApiKey,
				"key_01",
				json!({"key": "agw_sk_test", "metadata": {"agentgateway.dev/id": "key_01", "name": "test"}}),
			),
			test_resource(ConfigResourceKind::UiPolicy, "csrf", json!({})),
			test_resource(
				ConfigResourceKind::McpTarget,
				"db-target",
				json!({"name": "db-target", "stdio": {"cmd": "server"}}),
			),
			test_resource(
				ConfigResourceKind::McpPolicy,
				"cors",
				json!({"allowOrigins": ["https://example.com"]}),
			),
			test_resource(
				ConfigResourceKind::McpSettings,
				"default",
				json!({
					"statefulMode": "stateless",
					"prefixMode": "always",
					"failureMode": "failOpen"
				}),
			),
			test_resource(
				ConfigResourceKind::TrafficGateway,
				"public",
				json!({"name": "public", "port": 8080}),
			),
			test_resource(
				ConfigResourceKind::TrafficRoute,
				"later",
				json!({
					"name": "later",
					"gateways": ["public"],
					"backends": [{"host": "later.example:80"}]
				}),
			),
			test_resource(
				ConfigResourceKind::TrafficRoute,
				"earlier",
				json!({
					"name": "earlier",
					"gateways": ["public"],
					"backends": [{"host": "earlier.example:80"}]
				}),
			),
			test_resource(
				ConfigResourceKind::TrafficTcpRoute,
				"tcp",
				json!({
					"name": "tcp",
					"gateways": ["public"],
					"backends": [{"host": "tcp.example:80"}]
				}),
			),
		];

		let materialized = materialize_config(base, &resources).expect("materialize");
		let value: Value = crate::yaml::from_str(&materialized).expect("parse materialized");

		assert_eq!(
			value.pointer("/config/modelCatalog/0/inline/providers/database/models/database-model"),
			Some(&json!({}))
		);
		assert_eq!(
			value.pointer("/config/modelCatalog/1/inline/providers/file/models/file-model"),
			Some(&json!({}))
		);
		assert_eq!(
			value.pointer("/llm/providers/0/name"),
			Some(&json!("openai"))
		);
		assert_eq!(
			value.pointer("/llm/models/0/name"),
			Some(&json!("file-model"))
		);
		assert_eq!(
			value.pointer("/llm/policies/apiKey/keys/0/key"),
			Some(&json!("agw_sk_test"))
		);
		assert_eq!(
			value.pointer("/llm/policies/cors/allowOrigins/0"),
			Some(&json!("https://example.com"))
		);
		assert_eq!(value.pointer("/ui/policies/csrf"), Some(&json!({})));
		assert_eq!(
			value.pointer("/llm/models/1/name"),
			Some(&json!("db-model"))
		);
		assert_eq!(value.pointer("/llm/models/1/id"), Some(&json!("model_01")));
		assert_eq!(
			value.pointer("/llm/virtualModels/0/name"),
			Some(&json!("router"))
		);
		assert_eq!(
			value.pointer("/mcp/targets/0/name"),
			Some(&json!("file-target"))
		);
		assert_eq!(
			value.pointer("/mcp/targets/1/name"),
			Some(&json!("db-target"))
		);
		assert_eq!(
			value.pointer("/mcp/policies/cors/allowOrigins/0"),
			Some(&json!("https://example.com"))
		);
		assert_eq!(
			value.pointer("/mcp/statefulMode"),
			Some(&json!("stateless"))
		);
		assert_eq!(value.pointer("/mcp/prefixMode"), Some(&json!("always")));
		assert_eq!(value.pointer("/mcp/failureMode"), Some(&json!("failOpen")));
		assert_eq!(value.pointer("/gateways/public/port"), Some(&json!(8080)));
		assert_eq!(
			value.pointer("/gateways/public/name"),
			None,
			"resource identity must not leak into the gateway config"
		);
		assert_eq!(value.pointer("/routes/0/name"), Some(&json!("earlier")));
		assert_eq!(value.pointer("/routes/1/name"), Some(&json!("later")));
		assert_eq!(value.pointer("/tcpRoutes/0/name"), Some(&json!("tcp")));
	}

	#[test]
	fn materialization_rejects_file_owned_identity_conflicts() {
		let base = r#"
llm:
  providers:
  - name: openai
    provider: openAI
"#;
		let resources = vec![test_resource(
			ConfigResourceKind::LlmProvider,
			"openai",
			json!({"name": "openai", "provider": "openAI"}),
		)];

		let err = materialize_config(base, &resources).expect_err("conflict should fail");
		assert!(
			err
				.to_string()
				.contains("conflicts with file-owned resource"),
			"unexpected error: {err}"
		);

		let base = "llm:\n  models: []\n  policies:\n    cors: {}\n";
		let resources = vec![test_resource(
			ConfigResourceKind::LlmPolicy,
			"cors",
			json!({}),
		)];
		let err = materialize_config(base, &resources).expect_err("policy conflict should fail");
		assert!(
			err
				.to_string()
				.contains("config resource llm.policy/cors conflicts with file-owned resource"),
			"unexpected error: {err}"
		);

		let base = r#"
gateways:
  public:
    port: 8080
routes:
- name: api
  gateways: [public]
"#;
		let gateway_resources = vec![test_resource(
			ConfigResourceKind::TrafficGateway,
			"public",
			json!({"name": "public", "port": 8081}),
		)];
		let err =
			materialize_config(base, &gateway_resources).expect_err("gateway conflict should fail");
		assert!(
			err
				.to_string()
				.contains("config resource traffic.gateway/public conflicts with file-owned resource"),
			"unexpected error: {err}"
		);

		let route_resources = vec![test_resource(
			ConfigResourceKind::TrafficRoute,
			"api",
			json!({"name": "api", "gateways": ["public"]}),
		)];
		let err = materialize_config(base, &route_resources).expect_err("route conflict should fail");
		assert!(
			err
				.to_string()
				.contains("config resource traffic.route/api conflicts with file-owned resource"),
			"unexpected error: {err}"
		);
	}

	#[test]
	fn file_resource_writes_update_the_config_shape() {
		let mut config = json!({
			"config": {
				"modelCatalog": [
					{"inline": {"openai": {"gpt-file": {"input": 1.0}}}},
					{"file": "/tmp/base-costs.json"},
					{"inline": {"openai": {"gpt-file": {"output": 2.0}}}}
				]
			},
			"llm": {
				"models": [{
					"name": "file-model",
					"provider": "openai",
					"params": {"model": "gpt-file"}
				}],
				"policies": {
					"apiKey": {
						"mode": "strict",
						"keys": [{"key": "agw_sk_test", "metadata": {"name": "test"}}]
					}
				}
			},
			"mcp": {
				"port": 3000,
				"targets": [{"name": "tools", "mcp": {"host": "http://tools"}}]
			},
			"gateways": {
				"public": {"port": 8080},
				"private": {"port": 8081}
			},
			"routes": [
				{
					"name": "first",
					"gateways": ["public"],
					"backends": [{"host": "example.com:80"}]
				},
				{
					"name": "second",
					"gateways": ["public"],
					"backends": [{"host": "second.example.com:80"}]
				}
			]
		});

		let renamed = prepare_resource(
			ConfigResourceKind::LlmModel,
			json!({
				"name": "renamed-model",
				"provider": "openai",
				"params": {"model": "gpt-renamed"}
			}),
		)
		.expect("prepare model");
		upsert_file_config_resource(&mut config, &renamed, Some("file-model"))
			.expect("rename file model");
		assert_eq!(
			config.pointer("/llm/models/0/name"),
			Some(&json!("renamed-model"))
		);

		let colliding_route = prepare_resource(
			ConfigResourceKind::TrafficRoute,
			json!({"name": "second", "gateways": ["public"]}),
		)
		.expect("prepare colliding route");
		let err = upsert_file_config_resource(&mut config, &colliding_route, Some("first"))
			.expect_err("list rename collision must fail");
		assert!(err.to_string().contains("already exists"));
		assert_eq!(config.pointer("/routes/0/name"), Some(&json!("first")));

		let colliding_gateway = prepare_resource(
			ConfigResourceKind::TrafficGateway,
			json!({"name": "private", "port": 9090}),
		)
		.expect("prepare colliding gateway");
		let err = upsert_file_config_resource(&mut config, &colliding_gateway, Some("public"))
			.expect_err("map rename collision must fail");
		assert!(err.to_string().contains("already exists"));
		assert_eq!(config.pointer("/gateways/public/port"), Some(&json!(8080)));
		assert_eq!(config.pointer("/gateways/private/port"), Some(&json!(8081)));

		let missing_key =
			prepare_file_api_key_update("missing".to_string(), json!({"key": "agw_missing"}), None)
				.expect("prepare API key update");
		let err = upsert_file_config_resource(&mut config, &missing_key, Some("missing"))
			.expect_err("missing API key update must fail");
		assert!(err.to_string().contains("not found"));
		assert_eq!(
			config
				.pointer("/llm/policies/apiKey/keys")
				.and_then(Value::as_array)
				.map(Vec::len),
			Some(1)
		);

		assert!(
			delete_file_config_resource(&mut config, ConfigResourceKind::McpTarget, "tools")
				.expect("delete MCP target")
		);
		assert_eq!(config.pointer("/mcp/targets"), Some(&json!([])));

		let catalog = prepare_resource(
			ConfigResourceKind::ModelCatalog,
			json!({"custom": {"anthropic": {"claude": {"input": 3.0}}}}),
		)
		.expect("prepare model catalog");
		upsert_file_config_resource(&mut config, &catalog, None).expect("update file model catalog");
		assert_eq!(
			config.pointer("/config/modelCatalog"),
			Some(&json!([
				{"file": "/tmp/base-costs.json"},
				{"inline": {"anthropic": {"claude": {"input": 3.0}}}}
			]))
		);

		assert!(
			prepare_resource(
				ConfigResourceKind::TrafficRoute,
				json!({
					"gateways": ["public"],
					"backends": [{"host": "unnamed.example.com:80"}]
				}),
			)
			.is_err(),
			"traffic route writes require a name"
		);
		let route = prepare_resource(
			ConfigResourceKind::TrafficRoute,
			json!({
				"name": "updated",
				"gateways": ["public"],
				"backends": [{"host": "updated.example.com:80"}]
			}),
		)
		.expect("prepare route");
		upsert_file_config_resource(&mut config, &route, Some("first")).expect("update route");
		assert_eq!(config.pointer("/routes/0/name"), Some(&json!("updated")));
		assert_eq!(
			config.pointer("/routes/0/backends/0/host"),
			Some(&json!("updated.example.com:80"))
		);
		assert!(
			delete_file_config_resource(&mut config, ConfigResourceKind::TrafficRoute, &route.id,)
				.expect("delete route")
		);
		assert_eq!(config.pointer("/routes/0/name"), Some(&json!("second")));
	}
}
