PRAGMA foreign_keys = ON;

CREATE TABLE wm_schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

INSERT INTO wm_schema_version (version, applied_at)
VALUES (1, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));

CREATE TABLE wm_map_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    x REAL NOT NULL,
    y REAL NOT NULL,
    z REAL NOT NULL DEFAULT 0.0,
    kind TEXT NOT NULL CHECK(kind IN ('vendor','trainer','node','mob','waypoint','unknown')),
    discovered_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE wm_map_edges (
    from_id INTEGER NOT NULL REFERENCES wm_map_nodes(id) ON DELETE CASCADE,
    to_id INTEGER NOT NULL REFERENCES wm_map_nodes(id) ON DELETE CASCADE,
    cost REAL NOT NULL CHECK(cost >= 0.0),
    bidirectional INTEGER NOT NULL CHECK(bidirectional IN (0,1)),
    discovered_at TEXT NOT NULL,
    PRIMARY KEY (from_id, to_id)
);

CREATE TABLE wm_entities_seen (
    entity_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    last_x REAL NOT NULL,
    last_y REAL NOT NULL,
    last_z REAL NOT NULL DEFAULT 0.0,
    last_seen_at TEXT NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE wm_routes_taken (
    route_id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id INTEGER NOT NULL REFERENCES wm_map_nodes(id) ON DELETE CASCADE,
    to_id INTEGER NOT NULL REFERENCES wm_map_nodes(id) ON DELETE CASCADE,
    succeeded INTEGER NOT NULL CHECK(succeeded IN (0,1)),
    taken_at TEXT NOT NULL
);

CREATE TABLE wm_combat_history (
    combat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_entity_id TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK(outcome IN ('win','loss','flee','timeout','unknown')),
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL
);

CREATE INDEX idx_wm_map_nodes_kind ON wm_map_nodes(kind);
CREATE INDEX idx_wm_map_nodes_xy ON wm_map_nodes(x, y);
CREATE INDEX idx_wm_map_edges_from ON wm_map_edges(from_id);
CREATE INDEX idx_wm_map_edges_to ON wm_map_edges(to_id);
CREATE INDEX idx_wm_entities_seen_kind ON wm_entities_seen(kind);
CREATE INDEX idx_wm_routes_taken_from ON wm_routes_taken(from_id, taken_at);
CREATE INDEX idx_wm_combat_history_target ON wm_combat_history(target_entity_id);
