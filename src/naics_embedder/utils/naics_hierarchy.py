from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import polars as pl

class NaicsHierarchy:
    '''In-memory representation of the NAICS hierarchy derived from relations parquet data.'''

    def __init__(self, parent_child_pairs: Sequence[Tuple[str, str]]):
        self.parent_by_child: Dict[str, str] = {}
        self.children_by_parent = defaultdict(list)
        self._parent_child_pairs: List[Tuple[str, str]] = []

        seen_pairs = set()
        for parent, child in parent_child_pairs:
            if not parent or not child:
                continue
            key = (parent, child)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            # Keep the first observed parent for a child to avoid conflicting mappings.
            if child not in self.parent_by_child:
                self.parent_by_child[child] = parent
                self.children_by_parent[parent].append(child)
                self._parent_child_pairs.append(key)

    @classmethod
    def from_relations_parquet(cls, relations_path: Path) -> 'NaicsHierarchy':
        '''
        Build a hierarchy object from the relations parquet.

        Expects columns `code_i`, `code_j`, and either `relation_id` or `relation`/`relationship`.
        '''
        if not relations_path.exists():
            raise FileNotFoundError(f'NAICS relations parquet not found: {relations_path}')

        df = pl.read_parquet(relations_path)
        if 'code_i' not in df.columns or 'code_j' not in df.columns:
            raise ValueError('relations parquet must contain code_i and code_j columns')

        relation_expr = None
        if 'relation_id' in df.columns:
            relation_expr = pl.col('relation_id') == 1
        elif 'relation' in df.columns:
            relation_expr = pl.col('relation') == 'child'
        elif 'relationship' in df.columns:
            relation_expr = pl.col('relationship') == 'child'
        else:
            raise ValueError(
                'relations parquet must contain either relation_id or relation/relationship columns'
            )

        parent_child_pairs: List[Tuple[str, str]] = []
        for row in df.filter(relation_expr).select('code_i', 'code_j').iter_rows(named=True):
            parent = row['code_i']
            child = row['code_j']
            parent_child_pairs.append((parent, child))

        return cls(parent_child_pairs)

    def get_parent(self, code: str) -> Optional[str]:
        return self.parent_by_child.get(code)

    def get_children(self, code: str) -> List[str]:
        return list(self.children_by_parent.get(code, []))

    def get_siblings(self, code: str) -> List[str]:
        parent = self.get_parent(code)
        if parent is None:
            return []
        return [sibling for sibling in self.children_by_parent.get(parent, []) if sibling != code]

    @property
    def parent_child_pairs(self) -> List[Tuple[str, str]]:
        return list(self._parent_child_pairs)

@lru_cache(maxsize=4)
def load_naics_hierarchy(relations_path: str) -> NaicsHierarchy:
    '''Load (and cache) a NAICS hierarchy from the relations parquet file.'''

    path = Path(relations_path).expanduser().resolve()
    return NaicsHierarchy.from_relations_parquet(path)
