# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Structure-Aware Dynamic Curriculum (SADC) Scheduler
# -------------------------------------------------------------------------------------------------

class CurriculumScheduler:
    '''
    Structure-Aware Dynamic Curriculum (SADC) scheduler that manages training phases.
    
    Implements three-phase curriculum learning:
    1. Phase 1 (0-30%): Structural Initialization
       - Enforce symbolic tree constraints
       - Mask siblings
       - Weight negatives by inverse tree distance
    
    2. Phase 2 (30-70%): Geometric Refinement
       - Enable Lorentzian hard negative mining
       - Introduce Router-Guided sampling for MoE
    
    3. Phase 3 (70-100%): False Negative Mitigation
       - Activate clustering-based False Negative Elimination (FNE)
    '''
    
    def __init__(
        self,
        max_epochs: int,
        phase1_end: float = 0.3,
        phase2_end: float = 0.7,
        phase3_end: float = 1.0
    ):
        '''
        Initialize curriculum scheduler.
        
        Args:
            max_epochs: Maximum number of training epochs
            phase1_end: End of Phase 1 as fraction of max_epochs (default: 0.3)
            phase2_end: End of Phase 2 as fraction of max_epochs (default: 0.7)
            phase3_end: End of Phase 3 as fraction of max_epochs (default: 1.0)
        '''
        self.max_epochs = max_epochs
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.phase3_end = phase3_end
        
        # Phase boundaries in epochs
        self.phase1_end_epoch = int(max_epochs * phase1_end)
        self.phase2_end_epoch = int(max_epochs * phase2_end)
        self.phase3_end_epoch = int(max_epochs * phase3_end)
        
        logger.info(
            f'CurriculumScheduler initialized:\n'
            f'  Phase 1 (Structural Initialization): epochs 0-{self.phase1_end_epoch} (0-{phase1_end*100:.0f}%)\n'
            f'  Phase 2 (Geometric Refinement): epochs {self.phase1_end_epoch+1}-{self.phase2_end_epoch} ({phase1_end*100:.0f}-{phase2_end*100:.0f}%)\n'
            f'  Phase 3 (False Negative Mitigation): epochs {self.phase2_end_epoch+1}-{self.phase3_end_epoch} ({phase2_end*100:.0f}-{phase3_end*100:.0f}%)'
        )
    
    def get_phase(self, current_epoch: int) -> int:
        '''
        Get current phase based on epoch.
        
        Args:
            current_epoch: Current training epoch (0-indexed)
        
        Returns:
            Phase number (1, 2, or 3)
        '''
        if current_epoch <= self.phase1_end_epoch:
            return 1
        elif current_epoch <= self.phase2_end_epoch:
            return 2
        else:
            return 3
    
    def get_epoch_progress(self, current_epoch: int) -> float:
        '''
        Get training progress as fraction of max_epochs.
        
        Args:
            current_epoch: Current training epoch (0-indexed)
        
        Returns:
            Progress fraction in [0, 1]
        '''
        return min(current_epoch / self.max_epochs, 1.0)
    
    def get_curriculum_flags(self, current_epoch: int) -> Dict[str, bool]:
        '''
        Get curriculum flags for current epoch.
        
        Args:
            current_epoch: Current training epoch (0-indexed)
        
        Returns:
            Dictionary with curriculum flags:
            - use_tree_distance: Weight negatives by inverse tree distance
            - mask_siblings: Mask sibling negatives
            - enable_hard_negative_mining: Enable Lorentzian hard negative mining
            - enable_router_guided_sampling: Enable Router-Guided sampling for MoE
            - enable_clustering: Enable clustering-based FNE
        '''
        phase = self.get_phase(current_epoch)
        progress = self.get_epoch_progress(current_epoch)
        
        flags = {
            'use_tree_distance': phase == 1,  # Phase 1 only
            'mask_siblings': phase == 1,  # Phase 1 only
            'enable_hard_negative_mining': phase >= 2,  # Phase 2 and 3
            'enable_router_guided_sampling': phase >= 2,  # Phase 2 and 3
            'enable_clustering': phase >= 3,  # Phase 3 only
        }
        
        return flags
    
    def should_update_clustering(self, current_epoch: int, cluster_every_n_epochs: int) -> bool:
        '''
        Check if clustering should be updated based on curriculum phase.
        
        Args:
            current_epoch: Current training epoch (0-indexed)
            cluster_every_n_epochs: Update clustering every N epochs
        
        Returns:
            True if clustering should be updated
        '''
        phase = self.get_phase(current_epoch)
        if phase < 3:
            return False  # Only update in Phase 3
        
        # In Phase 3, check if it's time to update
        if current_epoch < self.phase2_end_epoch + 1:
            return False  # First epoch of Phase 3
        
        epochs_in_phase3 = current_epoch - self.phase2_end_epoch
        return epochs_in_phase3 % cluster_every_n_epochs == 0
    
    def log_phase_transition(self, current_epoch: int, previous_phase: Optional[int] = None):
        '''
        Log phase transition if epoch changed phase.
        
        Args:
            current_epoch: Current training epoch (0-indexed)
            previous_phase: Previous phase number (if known)
        '''
        current_phase = self.get_phase(current_epoch)
        
        if previous_phase is not None and current_phase != previous_phase:
            phase_names = {
                1: 'Structural Initialization',
                2: 'Geometric Refinement',
                3: 'False Negative Mitigation'
            }
            logger.info(
                f'\n{"="*80}\n'
                f'CURRICULUM PHASE TRANSITION: Phase {previous_phase} -> Phase {current_phase}\n'
                f'Phase {current_phase}: {phase_names[current_phase]}\n'
                f'Epoch: {current_epoch}/{self.max_epochs} ({self.get_epoch_progress(current_epoch)*100:.1f}%)\n'
                f'{"="*80}\n'
            )
            
            flags = self.get_curriculum_flags(current_epoch)
            logger.info('Curriculum flags:')
            for flag_name, flag_value in flags.items():
                logger.info(f'  • {flag_name}: {flag_value}')
            logger.info('')
    
    def get_negative_sample_weights(
        self,
        anchor_code: str,
        negative_codes: list,
        tree_distances: Optional[Dict] = None,
        relations: Optional[Dict] = None
    ) -> list:
        '''
        Get sampling weights for negative samples based on curriculum phase.
        
        Args:
            anchor_code: Anchor code
            negative_codes: List of negative codes
            tree_distances: Dictionary mapping (code1, code2) -> distance
            relations: Dictionary mapping (code1, code2) -> relation string
        
        Returns:
            List of sampling weights (normalized probabilities)
        '''
        # Default: uniform weights
        weights = [1.0] * len(negative_codes)
        
        # Phase 1: Weight by inverse tree distance
        # This is handled in the streaming dataset generator
        # Here we just return uniform weights as a placeholder
        # The actual weighting happens in the data generation layer
        
        return weights

