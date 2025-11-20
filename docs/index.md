# NAICS Hyperbolic Embedding System

This documentation describes a unified hyperbolic representation learning framework for the
**North American Industry Classification System (NAICS)**.  

The system consists of four sequential stages:

- Multi-channel transformer-based text encoding  
- Mixture-of-Experts fusion  
- Lorentz-model hyperbolic contrastive learning  
- Hyperbolic Graph Convolutional refinement (HGCN)

The final output are geometry-aware embeddings aligned with the hierarchical structure of
the NAICS taxonomy. These **Lorentz-model hyperbolic embeddings** are suitable for similarity
search, hierarchical modeling, graph-based reasoning, and downstream machine learning applications.

Use the navigation menu to explore system architecture, training procedures,
and API references for each module.
