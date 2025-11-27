'''
Unit tests for graph_model curriculum preprocessing.

Tests the difficulty annotation pipeline including node scoring,
relation cardinality classification, and threshold computation.
'''

import polars as pl
import pytest
import torch

from naics_embedder.graph_model.curriculum.preprocess_curriculum import (
    compute_degree_centrality,
    compute_difficulty_thresholds,
    compute_kcore,
    compute_node_scores,
    compute_pagerank,
    compute_relation_cardinality,
    load_distance_thresholds,
    load_node_scores,
    load_relation_types,
    preprocess_curriculum_data,
)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_graph_data(tmp_path):
    '''Create sample graph data for testing preprocessing.'''
    # Create descriptions parquet
    descriptions = pl.DataFrame(
        {
            'index': list(range(10)),
            'code':
            ['11', '111', '1111', '11111', '111111', '21', '211', '2111', '21111', '211111'],
            'level': [2, 3, 4, 5, 6, 2, 3, 4, 5, 6],
            'title': [f'Title {i}' for i in range(10)],
        }
    )
    desc_path = tmp_path / 'descriptions.parquet'
    descriptions.write_parquet(desc_path)

    # Create relations parquet
    relations = pl.DataFrame(
        {
            'idx_i': [0, 0, 1, 1, 2, 5, 5, 6, 6, 7],
            'idx_j': [1, 2, 2, 3, 4, 6, 7, 7, 8, 9],
            'code_i': ['11', '11', '111', '111', '1111', '21', '21', '211', '211', '2111'],
            'code_j': [
                '111',
                '1111',
                '1111',
                '11111',
                '111111',
                '211',
                '2111',
                '2111',
                '21111',
                '211111',
            ],
            'relation': [
                'child',
                'grandchild',
                'child',
                'grandchild',
                'child',
                'child',
                'grandchild',
                'child',
                'grandchild',
                'child',
            ],
            'relation_id': [1, 3, 1, 3, 1, 1, 3, 1, 3, 1],
        }
    )
    rel_path = tmp_path / 'relations.parquet'
    relations.write_parquet(rel_path)

    # Create distances parquet
    distances = pl.DataFrame(
        {
            'idx_i': [0, 0, 0, 1, 1, 2, 5, 5, 6],
            'idx_j': [1, 2, 3, 2, 3, 3, 6, 7, 7],
            'code_i': ['11', '11', '11', '111', '111', '1111', '21', '21', '211'],
            'code_j': ['111', '1111', '11111', '1111', '11111', '11111', '211', '2111', '2111'],
            'distance': [0.5, 1.5, 2.5, 0.5, 1.5, 0.5, 0.5, 1.5, 0.5],
        }
    )
    dist_path = tmp_path / 'distances.parquet'
    distances.write_parquet(dist_path)

    return {
        'descriptions': str(desc_path),
        'relations': str(rel_path),
        'distances': str(dist_path),
    }

@pytest.fixture
def sample_edge_index():
    '''Create sample edge index for centrality tests.'''
    # Simple tree structure
    src = torch.tensor([0, 0, 1, 1, 2, 5, 5, 6])
    dst = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    # Make bidirectional
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ])
    return edge_index

# -------------------------------------------------------------------------------------------------
# Centrality Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestDegreeCentrality:
    '''Test suite for degree centrality computation.'''

    def test_output_shape(self, sample_edge_index):
        '''Test output has correct shape.'''
        num_nodes = 10
        centrality = compute_degree_centrality(sample_edge_index, num_nodes)

        assert centrality.shape == (num_nodes, )

    def test_values_in_range(self, sample_edge_index):
        '''Test centrality values are in [0, 1].'''
        num_nodes = 10
        centrality = compute_degree_centrality(sample_edge_index, num_nodes)

        assert torch.all(centrality >= 0)
        assert torch.all(centrality <= 1)

    def test_hub_has_higher_centrality(self, sample_edge_index):
        '''Test hub nodes have higher centrality.'''
        num_nodes = 10
        centrality = compute_degree_centrality(sample_edge_index, num_nodes)

        # Node 0 and 1 are hubs (more connections)
        # Node 8 is a leaf
        assert centrality[0] > centrality[8] or centrality[1] > centrality[8]

@pytest.mark.unit
class TestPageRank:
    '''Test suite for PageRank computation.'''

    def test_output_shape(self, sample_edge_index):
        '''Test output has correct shape.'''
        num_nodes = 10
        pagerank = compute_pagerank(sample_edge_index, num_nodes)

        assert pagerank.shape == (num_nodes, )

    def test_sums_to_one(self, sample_edge_index):
        '''Test PageRank values sum to 1.'''
        num_nodes = 10
        pagerank = compute_pagerank(sample_edge_index, num_nodes)

        assert torch.allclose(pagerank.sum(), torch.tensor(1.0), atol=1e-5)

    def test_all_positive(self, sample_edge_index):
        '''Test all PageRank values are positive.'''
        num_nodes = 10
        pagerank = compute_pagerank(sample_edge_index, num_nodes)

        assert torch.all(pagerank > 0)

@pytest.mark.unit
class TestKCore:
    '''Test suite for k-core decomposition.'''

    def test_output_shape(self, sample_edge_index):
        '''Test output has correct shape.'''
        num_nodes = 10
        kcore = compute_kcore(sample_edge_index, num_nodes)

        assert kcore.shape == (num_nodes, )

    def test_non_negative(self, sample_edge_index):
        '''Test k-core values are non-negative.'''
        num_nodes = 10
        kcore = compute_kcore(sample_edge_index, num_nodes)

        assert torch.all(kcore >= 0)

# -------------------------------------------------------------------------------------------------
# Node Scores Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeNodeScores:
    '''Test suite for compute_node_scores function.'''

    def test_returns_all_metrics(self, sample_graph_data, tmp_path):
        '''Test all expected metrics are returned.'''
        output_path = str(tmp_path / 'node_scores.pt')

        scores = compute_node_scores(
            sample_graph_data['descriptions'],
            sample_graph_data['relations'],
            output_path=output_path,
        )

        assert 'degree_centrality' in scores
        assert 'pagerank' in scores
        assert 'kcore' in scores
        assert 'level' in scores
        assert 'composite' in scores
        assert 'num_nodes' in scores

    def test_saves_to_file(self, sample_graph_data, tmp_path):
        '''Test scores are saved to file.'''
        output_path = tmp_path / 'node_scores.pt'

        compute_node_scores(
            sample_graph_data['descriptions'],
            sample_graph_data['relations'],
            output_path=str(output_path),
        )

        assert output_path.exists()

    def test_load_saved_scores(self, sample_graph_data, tmp_path):
        '''Test saved scores can be loaded.'''
        output_path = str(tmp_path / 'node_scores.pt')

        original = compute_node_scores(
            sample_graph_data['descriptions'],
            sample_graph_data['relations'],
            output_path=output_path,
        )

        loaded = load_node_scores(output_path)

        assert torch.allclose(original['composite'], loaded['composite'])

# -------------------------------------------------------------------------------------------------
# Relation Cardinality Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeRelationCardinality:
    '''Test suite for relation cardinality classification.'''

    def test_returns_all_fields(self, sample_graph_data, tmp_path):
        '''Test all expected fields are returned.'''
        output_path = str(tmp_path / 'relation_types.json')

        result = compute_relation_cardinality(
            sample_graph_data['relations'],
            output_path=output_path,
        )

        assert 'relation_types' in result
        assert 'relation_difficulty' in result
        assert 'statistics' in result

    def test_classifies_relations(self, sample_graph_data, tmp_path):
        '''Test relations are classified correctly.'''
        result = compute_relation_cardinality(sample_graph_data['relations'], )

        # Check child relation is classified
        assert 'child' in result['relation_types']
        cardinality = result['relation_types']['child']
        assert cardinality in ['1-1', '1-N', 'N-1', 'N-N']

    def test_difficulty_tiers(self, sample_graph_data, tmp_path):
        '''Test difficulty tiers are assigned.'''
        result = compute_relation_cardinality(sample_graph_data['relations'], )

        # relation_id 1 (child) should be easy
        assert result['relation_difficulty'][1] == 'easy'

        # relation_id 3 (grandchild) should be medium
        assert result['relation_difficulty'][3] == 'medium'

    def test_saves_to_file(self, sample_graph_data, tmp_path):
        '''Test result is saved to file.'''
        output_path = tmp_path / 'relation_types.json'

        compute_relation_cardinality(
            sample_graph_data['relations'],
            output_path=str(output_path),
        )

        assert output_path.exists()

    def test_load_saved_types(self, sample_graph_data, tmp_path):
        '''Test saved types can be loaded.'''
        output_path = str(tmp_path / 'relation_types.json')

        original = compute_relation_cardinality(
            sample_graph_data['relations'],
            output_path=output_path,
        )

        loaded = load_relation_types(output_path)

        assert original['relation_types'] == loaded['relation_types']

# -------------------------------------------------------------------------------------------------
# Difficulty Thresholds Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeDifficultyThresholds:
    '''Test suite for difficulty threshold computation.'''

    def test_returns_phase_thresholds(self, sample_graph_data, tmp_path):
        '''Test phase thresholds are returned.'''
        # Create minimal triplets for testing
        triplets_dir = tmp_path / 'triplets'
        triplets_dir.mkdir()

        result = compute_difficulty_thresholds(
            sample_graph_data['distances'],
            str(triplets_dir),
        )

        assert 'phase1_max_distance' in result
        assert 'phase2_max_distance' in result
        assert 'phase3_max_distance' in result
        assert 'phase4_max_distance' in result

    def test_thresholds_ordered(self, sample_graph_data, tmp_path):
        '''Test thresholds are in ascending order.'''
        triplets_dir = tmp_path / 'triplets'
        triplets_dir.mkdir()

        result = compute_difficulty_thresholds(
            sample_graph_data['distances'],
            str(triplets_dir),
        )

        assert result['phase1_max_distance'] <= result['phase2_max_distance']
        assert result['phase2_max_distance'] <= result['phase3_max_distance']
        assert result['phase3_max_distance'] <= result['phase4_max_distance']

    def test_distance_statistics(self, sample_graph_data, tmp_path):
        '''Test distance statistics are included.'''
        triplets_dir = tmp_path / 'triplets'
        triplets_dir.mkdir()

        result = compute_difficulty_thresholds(
            sample_graph_data['distances'],
            str(triplets_dir),
        )

        assert 'distance_statistics' in result
        stats = result['distance_statistics']
        assert 'min' in stats
        assert 'max' in stats
        assert 'mean' in stats

# -------------------------------------------------------------------------------------------------
# Full Pipeline Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestPreprocessCurriculumData:
    '''Test suite for full preprocessing pipeline.'''

    def test_full_pipeline(self, sample_graph_data, tmp_path):
        '''Test full preprocessing pipeline runs successfully.'''
        output_dir = tmp_path / 'curriculum_cache'

        # Create empty triplets directory
        triplets_dir = tmp_path / 'triplets'
        triplets_dir.mkdir()

        node_scores, relation_types, thresholds = preprocess_curriculum_data(
            descriptions_parquet=sample_graph_data['descriptions'],
            relations_parquet=sample_graph_data['relations'],
            distances_parquet=sample_graph_data['distances'],
            triplets_parquet=str(triplets_dir),
            output_dir=str(output_dir),
        )

        # Check all outputs
        assert 'composite' in node_scores
        assert 'relation_types' in relation_types
        assert 'phase1_max_distance' in thresholds

    def test_creates_output_files(self, sample_graph_data, tmp_path):
        '''Test output files are created.'''
        output_dir = tmp_path / 'curriculum_cache'
        triplets_dir = tmp_path / 'triplets'
        triplets_dir.mkdir()

        preprocess_curriculum_data(
            descriptions_parquet=sample_graph_data['descriptions'],
            relations_parquet=sample_graph_data['relations'],
            distances_parquet=sample_graph_data['distances'],
            triplets_parquet=str(triplets_dir),
            output_dir=str(output_dir),
        )

        assert (output_dir / 'node_scores.pt').exists()
        assert (output_dir / 'relation_types.json').exists()
        assert (output_dir / 'difficulty_thresholds.json').exists()

    def test_output_files_loadable(self, sample_graph_data, tmp_path):
        '''Test output files can be loaded.'''
        output_dir = tmp_path / 'curriculum_cache'
        triplets_dir = tmp_path / 'triplets'
        triplets_dir.mkdir()

        preprocess_curriculum_data(
            descriptions_parquet=sample_graph_data['descriptions'],
            relations_parquet=sample_graph_data['relations'],
            distances_parquet=sample_graph_data['distances'],
            triplets_parquet=str(triplets_dir),
            output_dir=str(output_dir),
        )

        # Load all outputs
        node_scores = load_node_scores(str(output_dir / 'node_scores.pt'))
        relation_types = load_relation_types(str(output_dir / 'relation_types.json'))
        thresholds = load_distance_thresholds(str(output_dir / 'difficulty_thresholds.json'))

        assert 'composite' in node_scores
        assert 'relation_types' in relation_types
        assert 'phase1_max_distance' in thresholds
