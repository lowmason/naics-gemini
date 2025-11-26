'''
Unit tests for NAICS distance computation.

Tests tree construction, distance calculations, and relationship computations.
'''

import networkx as nx
import polars as pl
import pytest

from naics_embedder.data.compute_distances import (
    _compute_tree_metadata,
    _find_common_ancestor,
    _get_distance,
    _join_sectors,
    _sector_codes,
    _sector_tree,
    _sectors,
)

# -------------------------------------------------------------------------------------------------
# Sector Utilities Tests
# -------------------------------------------------------------------------------------------------


@pytest.mark.unit
class TestSectorUtilities:
    '''Test suite for sector utility functions.'''

    def test_join_sectors_manufacturing(self):
        '''Test joining of manufacturing sectors (31-33).'''

        assert _join_sectors('31') == '31'
        assert _join_sectors('32') == '31'
        assert _join_sectors('33') == '31'

    def test_join_sectors_retail(self):
        '''Test joining of retail sectors (44-45).'''

        assert _join_sectors('44') == '44'
        assert _join_sectors('45') == '44'

    def test_join_sectors_transportation(self):
        '''Test joining of transportation sectors (48-49).'''

        assert _join_sectors('48') == '48'
        assert _join_sectors('49') == '48'

    def test_join_sectors_unchanged(self):
        '''Test that other sectors remain unchanged.'''

        assert _join_sectors('11') == '11'
        assert _join_sectors('22') == '22'
        assert _join_sectors('51') == '51'
        assert _join_sectors('62') == '62'

    def test_sectors_extraction(self, sample_naics_data):
        '''Test extraction of level-2 sectors.'''

        sectors = _sectors(sample_naics_data)

        # Should extract unique level-2 codes
        assert isinstance(sectors, list)
        assert len(sectors) > 0
        assert all(len(code) == 2 for code in sectors)

    def test_sector_codes_extraction(self, sample_naics_data):
        '''Test extraction of level-6 codes for a sector.'''

        codes = _sector_codes('31', sample_naics_data)

        # Should extract level-6 codes for sector 31
        assert isinstance(codes, list)
        assert all(len(code) == 6 for code in codes)
        assert all(code.startswith(('31', '32', '33')) for code in codes)


# -------------------------------------------------------------------------------------------------
# Tree Construction Tests
# -------------------------------------------------------------------------------------------------


@pytest.mark.unit
class TestTreeConstruction:
    '''Test suite for NAICS tree construction.'''

    def test_sector_tree_structure(self, sample_naics_data):
        '''Test that sector tree has correct structure.'''

        tree = _sector_tree('31', sample_naics_data)

        assert isinstance(tree, nx.DiGraph)
        assert tree.number_of_nodes() > 0

    def test_sector_tree_hierarchy(self, sample_naics_data):
        '''Test that tree follows hierarchical structure.'''

        tree = _sector_tree('31', sample_naics_data)

        # Parent-child relationships should exist
        # e.g., 31 -> 311 -> 3111 -> 31111 -> 311111
        if tree.number_of_nodes() >= 2:
            # Check that edges exist (specific codes depend on sample data)
            assert tree.number_of_edges() > 0

    def test_tree_is_directed(self, sample_naics_data):
        '''Test that tree is a directed graph.'''

        tree = _sector_tree('31', sample_naics_data)

        assert isinstance(tree, nx.DiGraph)

    def test_tree_no_cycles(self, sample_naics_data):
        '''Test that tree has no cycles (is a proper tree).'''

        tree = _sector_tree('31', sample_naics_data)

        # A tree should be acyclic
        assert nx.is_directed_acyclic_graph(tree)


# -------------------------------------------------------------------------------------------------
# Tree Metadata Tests
# -------------------------------------------------------------------------------------------------


@pytest.mark.unit
class TestTreeMetadata:
    '''Test suite for tree metadata computation.'''

    def test_compute_metadata_returns_dicts(self, sample_naics_data):
        '''Test that metadata computation returns dictionaries.'''

        tree = _sector_tree('31', sample_naics_data)
        depths, ancestors, parents = _compute_tree_metadata(tree, '31')

        assert isinstance(depths, dict)
        assert isinstance(ancestors, dict)
        assert isinstance(parents, dict)

    def test_root_depth_zero(self, sample_naics_data):
        '''Test that root node has depth 0.'''

        tree = _sector_tree('31', sample_naics_data)
        depths, _, _ = _compute_tree_metadata(tree, '31')

        assert '31' in depths
        assert depths['31'] == 0

    def test_root_has_no_parent(self, sample_naics_data):
        '''Test that root node has no parent.'''

        tree = _sector_tree('31', sample_naics_data)
        _, _, parents = _compute_tree_metadata(tree, '31')

        assert '31' in parents
        assert parents['31'] is None

    def test_root_in_own_ancestors(self, sample_naics_data):
        '''Test that root is in its own ancestor list.'''

        tree = _sector_tree('31', sample_naics_data)
        _, ancestors, _ = _compute_tree_metadata(tree, '31')

        assert '31' in ancestors
        assert '31' in ancestors['31']

    def test_depth_increases_with_level(self, sample_naics_data):
        '''Test that depth increases for child nodes.'''

        tree = _sector_tree('31', sample_naics_data)
        depths, ancestors, parents = _compute_tree_metadata(tree, '31')

        # Find a node with a parent
        for node, parent in parents.items():
            if parent is not None:
                assert depths[node] == depths[parent] + 1
                break

    def test_ancestors_include_parents(self, sample_naics_data):
        '''Test that ancestors list includes all parents.'''

        tree = _sector_tree('31', sample_naics_data)
        depths, ancestors, parents = _compute_tree_metadata(tree, '31')

        # For any node with a parent, parent should be in ancestors
        for node, parent in parents.items():
            if parent is not None:
                assert parent in ancestors[node]


# -------------------------------------------------------------------------------------------------
# Common Ancestor Tests
# -------------------------------------------------------------------------------------------------


@pytest.mark.unit
class TestCommonAncestor:
    '''Test suite for common ancestor finding.'''

    def test_find_common_ancestor_siblings(self, sample_naics_data):
        '''Test finding common ancestor for sibling nodes.'''

        tree = _sector_tree('31', sample_naics_data)
        _, ancestors, _ = _compute_tree_metadata(tree, '31')

        # Get two nodes if they exist
        nodes = list(ancestors.keys())
        if len(nodes) >= 2:
            common = _find_common_ancestor(nodes[0], nodes[1], ancestors)
            assert common is not None
            assert common in ancestors[nodes[0]]
            assert common in ancestors[nodes[1]]

    def test_find_common_ancestor_self(self, sample_naics_data):
        '''Test that node's common ancestor with itself is itself.'''

        tree = _sector_tree('31', sample_naics_data)
        _, ancestors, _ = _compute_tree_metadata(tree, '31')

        node = '31'
        common = _find_common_ancestor(node, node, ancestors)

        assert common == node

    def test_find_common_ancestor_parent_child(self, sample_naics_data):
        '''Test common ancestor for parent-child relationship.'''

        tree = _sector_tree('31', sample_naics_data)
        _, ancestors, parents = _compute_tree_metadata(tree, '31')

        # Find a parent-child pair
        for child, parent in parents.items():
            if parent is not None:
                common = _find_common_ancestor(parent, child, ancestors)
                # Common ancestor should be the parent
                assert common == parent
                break


# -------------------------------------------------------------------------------------------------
# Distance Computation Tests
# -------------------------------------------------------------------------------------------------


@pytest.mark.unit
class TestDistanceComputation:
    '''Test suite for distance calculations.'''

    def test_distance_to_self_is_zero(self, sample_naics_data):
        '''Test that distance from a node to itself is 0.'''

        tree = _sector_tree('31', sample_naics_data)
        depths, ancestors, _ = _compute_tree_metadata(tree, '31')

        node = '31'
        distance = _get_distance(node, node, depths, ancestors)

        assert distance == 0.0

    def test_distance_is_symmetric(self, sample_naics_data):
        '''Test that distance(i, j) == distance(j, i).'''

        tree = _sector_tree('31', sample_naics_data)
        depths, ancestors, _ = _compute_tree_metadata(tree, '31')

        nodes = list(depths.keys())
        if len(nodes) >= 2:
            node1, node2 = nodes[0], nodes[1]

            d_12 = _get_distance(node1, node2, depths, ancestors)
            d_21 = _get_distance(node2, node1, depths, ancestors)

            assert d_12 == d_21

    def test_lineal_distance_symmetric(self, sample_naics_data):
        '''Test ancestor/descendant relationships are symmetric.'''

        tree = _sector_tree('31', sample_naics_data)
        depths, ancestors, parents = _compute_tree_metadata(tree, '31')

        for child, parent in parents.items():
            if parent is not None:
                d_pc = _get_distance(parent, child, depths, ancestors)
                d_cp = _get_distance(child, parent, depths, ancestors)

                assert d_pc == d_cp == 0.5
                break

    def test_distance_parent_child(self, sample_naics_data):
        '''Test distance between parent and child.'''

        tree = _sector_tree('31', sample_naics_data)
        depths, ancestors, parents = _compute_tree_metadata(tree, '31')

        # Find a parent-child pair
        for child, parent in parents.items():
            if parent is not None:
                distance = _get_distance(parent, child, depths, ancestors)

                # Direct parent-child should have distance 0.5 (lineal relationship)
                assert distance == 0.5
                break

    def test_distance_increases_with_separation(self, sample_naics_data):
        '''Test that distance increases with tree separation.'''

        tree = _sector_tree('31', sample_naics_data)
        depths, ancestors, parents = _compute_tree_metadata(tree, '31')

        # Root to immediate child
        root = '31'
        immediate_children = [node for node, parent in parents.items() if parent == root]

        if immediate_children:
            child = immediate_children[0]

            # Distance to immediate child
            d_immediate = _get_distance(root, child, depths, ancestors)

            # Find grandchild if exists
            grandchildren = [node for node, parent in parents.items() if parent == child]

            if grandchildren:
                grandchild = grandchildren[0]
                d_grand = _get_distance(root, grandchild, depths, ancestors)

                # Distance to grandchild should be greater
                assert d_grand > d_immediate

    def test_distance_non_negative(self, sample_naics_data):
        '''Test that all distances are non-negative.'''

        tree = _sector_tree('31', sample_naics_data)
        depths, ancestors, _ = _compute_tree_metadata(tree, '31')

        nodes = list(depths.keys())

        # Test several pairs
        for i in range(min(5, len(nodes))):
            for j in range(i, min(5, len(nodes))):
                distance = _get_distance(nodes[i], nodes[j], depths, ancestors)
                assert distance >= 0.0


# -------------------------------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------------------------------


@pytest.mark.unit
class TestDistanceIntegration:
    '''Integration tests for complete distance computation workflow.'''

    def test_full_sector_distance_computation(self, sample_naics_data):
        '''Test complete distance computation for a sector.'''

        # Get sectors
        sectors = _sectors(sample_naics_data)

        if sectors:
            sector = sectors[0]

            # Build tree
            tree = _sector_tree(sector, sample_naics_data)

            # Compute metadata
            depths, ancestors, parents = _compute_tree_metadata(tree, sector)

            # Compute distances for all pairs
            nodes = list(depths.keys())
            distances = {}

            for i, node_i in enumerate(nodes):
                for node_j in nodes[i:]:
                    d = _get_distance(node_i, node_j, depths, ancestors)
                    distances[(node_i, node_j)] = d

            # Verify all distances computed
            assert len(distances) > 0

            # Verify all distances are valid
            for d in distances.values():
                assert isinstance(d, (int, float))
                assert d >= 0.0

    def test_tree_properties_preserved(self, sample_naics_data):
        '''Test that computed distances preserve tree properties.'''

        tree = _sector_tree('31', sample_naics_data)
        depths, ancestors, _ = _compute_tree_metadata(tree, '31')

        nodes = list(depths.keys())

        if len(nodes) >= 3:
            # Test triangle inequality for a few triplets
            for i in range(min(3, len(nodes) - 2)):
                node_a = nodes[i]
                node_b = nodes[i + 1]
                node_c = nodes[i + 2]

                d_ab = _get_distance(node_a, node_b, depths, ancestors)
                d_bc = _get_distance(node_b, node_c, depths, ancestors)
                d_ac = _get_distance(node_a, node_c, depths, ancestors)

                # Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
                # Allow small tolerance for lineal adjustments
                assert d_ac <= d_ab + d_bc + 1.0


# -------------------------------------------------------------------------------------------------
# Edge Cases
# -------------------------------------------------------------------------------------------------


@pytest.mark.unit
class TestDistanceEdgeCases:
    '''Test suite for edge cases in distance computation.'''

    def test_single_node_tree(self, tmp_path):
        '''Test distance computation for single-node tree.'''

        # Create minimal data with just one code
        data = {
            'index': [0],
            'code': ['11'],
            'level': [2],
            'title': ['Agriculture'],
        }

        df = pl.DataFrame(data)
        path = tmp_path / 'single_node.parquet'
        df.write_parquet(path)

        tree = _sector_tree('11', str(path))
        depths, ancestors, _ = _compute_tree_metadata(tree, '11')

        # Distance to self should be 0
        d = _get_distance('11', '11', depths, ancestors)
        assert d == 0.0

    def test_two_node_tree(self, tmp_path):
        '''Test distance computation for two-node tree.'''

        data = {
            'index': [0, 1],
            'code': ['11', '111'],
            'level': [2, 3],
            'title': ['Agriculture', 'Crop Production'],
        }

        df = pl.DataFrame(data)
        path = tmp_path / 'two_node.parquet'
        df.write_parquet(path)

        tree = _sector_tree('11', str(path))
        depths, ancestors, _ = _compute_tree_metadata(tree, '11')

        # Parent-child distance
        d = _get_distance('11', '111', depths, ancestors)
        assert d == 0.5  # Lineal relationship
