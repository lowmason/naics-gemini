# -------------------------------------------------------------------------------------------------
# Event Bus for Curriculum Learning (#52)
# -------------------------------------------------------------------------------------------------
'''
Lightweight pub/sub event bus for decoupled communication between curriculum components.

Allows the CurriculumController to notify listeners (DataModule sampler, loss module,
trainer hooks) without tight coupling.
'''

import logging
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

class CurriculumEvent(Enum):
    '''Events that can be published on the curriculum event bus.'''

    # Phase transitions
    PHASE_ADVANCE = auto()  # Controller advanced to next phase
    PHASE_ROLLBACK = auto()  # Controller rolled back to previous phase

    # Sampling updates
    SAMPLER_UPDATE = auto()  # Sampler parameters changed
    DIFFICULTY_THRESHOLD_UPDATE = auto()  # Difficulty thresholds changed

    # Loss updates
    MARGIN_UPDATE = auto()  # Loss margin changed
    TEMPERATURE_UPDATE = auto()  # Temperature changed
    CURVATURE_UPDATE = auto()  # Curvature changed

    # Training signals
    PLATEAU_DETECTED = auto()  # Loss plateau detected
    OVERFITTING_DETECTED = auto()  # Generalization gap too large
    COLLAPSE_DETECTED = auto()  # Representation collapse detected

    # Checkpointing
    CHECKPOINT_SAVE = auto()  # Save rollback checkpoint
    CHECKPOINT_RESTORE = auto()  # Restore from rollback checkpoint

    # Metrics
    METRICS_LOGGED = auto()  # New metrics available

# Type alias for event handlers
EventHandler = Callable[[CurriculumEvent, Dict[str, Any]], None]

class EventBus:
    """
    Simple observer pattern event bus for curriculum learning.

    Components register handlers for specific events, and the controller
    publishes events when state changes occur.

    Example:
        bus = EventBus()

        def on_phase_advance(event, data):
            print(f"Advanced to phase {data['new_phase']}")

        bus.subscribe(CurriculumEvent.PHASE_ADVANCE, on_phase_advance)
        bus.publish(CurriculumEvent.PHASE_ADVANCE, {'new_phase': 2})
    """

    def __init__(self) -> None:
        self._handlers: Dict[CurriculumEvent, List[EventHandler]] = defaultdict(list)
        self._global_handlers: List[EventHandler] = []
        self._event_history: List[Dict[str, Any]] = []
        self._max_history: int = 1000

    def subscribe(
        self,
        event: CurriculumEvent,
        handler: EventHandler,
    ) -> None:
        '''
        Subscribe a handler to a specific event type.

        Args:
            event: The event type to subscribe to
            handler: Callable that receives (event, data) when event fires
        '''
        if handler not in self._handlers[event]:
            self._handlers[event].append(handler)
            logger.debug(f'Subscribed handler to {event.name}')

    def subscribe_all(self, handler: EventHandler) -> None:
        '''
        Subscribe a handler to all events (useful for logging/monitoring).

        Args:
            handler: Callable that receives (event, data) for all events
        '''
        if handler not in self._global_handlers:
            self._global_handlers.append(handler)
            logger.debug('Subscribed global handler')

    def unsubscribe(
        self,
        event: CurriculumEvent,
        handler: EventHandler,
    ) -> None:
        '''
        Unsubscribe a handler from a specific event type.

        Args:
            event: The event type to unsubscribe from
            handler: The handler to remove
        '''
        if handler in self._handlers[event]:
            self._handlers[event].remove(handler)
            logger.debug(f'Unsubscribed handler from {event.name}')

    def unsubscribe_all(self, handler: EventHandler) -> None:
        '''Remove a global handler.'''
        if handler in self._global_handlers:
            self._global_handlers.remove(handler)

    def publish(
        self,
        event: CurriculumEvent,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        '''
        Publish an event to all subscribed handlers.

        Args:
            event: The event type to publish
            data: Optional data payload for the event
        '''
        data = data or {}

        # Record in history
        self._event_history.append({
            'event': event.name,
            'data': data,
        })

        # Trim history if needed
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Notify specific handlers
        for handler in self._handlers[event]:
            try:
                handler(event, data)
            except Exception as e:
                logger.error(f'Error in handler for {event.name}: {e}')

        # Notify global handlers
        for handler in self._global_handlers:
            try:
                handler(event, data)
            except Exception as e:
                logger.error(f'Error in global handler for {event.name}: {e}')

    def get_history(
        self,
        event_filter: Optional[CurriculumEvent] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        '''
        Get recent event history, optionally filtered by event type.

        Args:
            event_filter: Optional event type to filter by
            limit: Maximum number of events to return

        Returns:
            List of event records (most recent last)
        '''
        if event_filter is not None:
            filtered = [e for e in self._event_history if e['event'] == event_filter.name]
            return filtered[-limit:]
        return self._event_history[-limit:]

    def clear_history(self) -> None:
        '''Clear the event history.'''
        self._event_history.clear()

    def clear_handlers(self) -> None:
        '''Clear all registered handlers.'''
        self._handlers.clear()
        self._global_handlers.clear()

# -------------------------------------------------------------------------------------------------
# Global Event Bus Instance
# -------------------------------------------------------------------------------------------------

# Singleton instance for convenience (can also instantiate directly)
_global_bus: Optional[EventBus] = None

def get_event_bus() -> EventBus:
    '''Get or create the global event bus instance.'''
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus

def reset_event_bus() -> None:
    '''Reset the global event bus (useful for testing).'''
    global _global_bus
    if _global_bus is not None:
        _global_bus.clear_handlers()
        _global_bus.clear_history()
    _global_bus = None
