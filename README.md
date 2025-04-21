# Adaptation of reticula for mouse toxicology expression data

## Reference Repository
This repository is based on the work presented in the following repository:

[reticula](https://zenodo.org/badge/latestdoi/212217385)

Accompanying paper:

[Burkhart JG, Wu G, Song X, Raimondi F, McWeeney S, Wong MH, Deng Y. Biology-inspired graph neural network encodes reactome and reveals biochemical reactions of disease. Patterns. 2023 May 22.](https://doi.org/10.1016/j.patter.2023.100758)

The original repository contains workflows and scripts to process human gene expression data and map it onto a biochemical reaction network for graph neural network (GNN) classification. This repository adapts that workflow for **mouse gene expression data**, necessitating changes in several key input files.

## Key Differences
- **Input Data:** The original repository uses human gene expression data from GTEx, TCGA, and SRP035988. Our adaptation works with **mouse gene expression datasets** obtained from [Recount3](https://rna.recount.bio/) derived from datasets available on the Gene Expression Omnibus [(GEO)](https://www.ncbi.nlm.nih.gov/geo/) and [(ArrayExpress)](https://www.ebi.ac.uk/biostudies/arrayexpress).
- **Reactome Mapping:** We use mouse-specific Reactome reaction networks and gene-to-reaction mappings instead of the human equivalents.
- **File Modifications:** Several processing scripts were updated to reflect these differences, ensuring proper preprocessing, feature extraction, and model training.
- **Reactome Data Extraction:** To generate mouse-specific Reactome mappings, we:
  1. **Produce `ReactionNetwork_Rel.txt`**
     - Install Docker.
     - Run the Reactome graph database using:
       ```
       docker run -p 7474:7474 -p 7687:7687 -e NEO4J_dbms_memory_heap_maxSize=8g reactome/graphdb:latest
       ```
     - The username is **Neo4j** and the password is **admin**.
     - Use the following queries to extract reaction relationships:
       ```
       MATCH (r2:ReactionLikeEvent {speciesName:"Mus musculus"})-[:precedingEvent]->(r1:ReactionLikeEvent {speciesName:"Mus musculus"}) RETURN r1.stId as Preceding Reaction, r2.stId as Following Reaction
       ```
       ```
       MATCH (r1:ReactionLikeEvent {speciesName:"Mus musculus"})-[:output]->(PhysicalEntity)<-[:physicalEntity]-(CatalystActivity)<-[:catalystActivity]-(r2:ReactionLikeEvent {speciesName:"Mus musculus"}) RETURN r1.stId as Preceding Reaction, r2.stId as Following Reaction
       ```
     - Download the results as `Mouse1.csv` and `Mouse2.csv`, then integrate them.
  2. **Produce `ReactionToPathway_Rel.csv`**
     - Run the Reactome graph database as above.
     - Use the following query to retrieve pathway-reaction relationships:
       ```
       MATCH (p:Pathway {speciesName:"Mus musculus"})-[:hasEvent]->(r:ReactionLikeEvent) RETURN p.stId as Pathway, r.stId as ReactionLikeEvent, r.displayName as Title
       ```
     - Download the results as `ReactionToPathway_Rel.csv`.
- **Model Training:** We follow the core methodology but retrain the GNN with mouse-specific data to assess biochemical reaction patterns in our new study.

### Graphical representation of key changes
![Graphical Abstract](assets/graphical_abstract.png)

## Associated Paper
Our application of this modified workflow is detailed in:

**Application of a metabolic network-based graph neural network for the identification of toxicant-induced perturbations**  
Yuan, K., and Nault, R.  
[Journal or Preprint Link]  

This paper extends the original methodology to investigate **[specific focus of your study]**, demonstrating its applicability to mouse gene expression datasets.

## Repository Structure
The organization of this repository follows the structure of the original repository, with adjustments to input files and modified processing scripts. Below is a general overview:

- `data/` - Mouse-specific gene expression datasets and Reactome mappings  
- `src/r/` - R scripts adapted for processing mouse data  
- `src/python/` - Python scripts for training and validating the GNN model  
- `notebooks/` - Jupyter notebooks for exploratory analysis and model training  

## Citation
If you use this repository, please cite both our paper and the original Reticula paper:

**Original Reticula Paper:**  
Burkhart JG, Wu G, Song X, et al. *Biology-inspired graph neural network encodes reactome and reveals biochemical reactions of disease.* Patterns. 2023 May 22. [DOI: 10.1016/j.patter.2023.100758](https://doi.org/10.1016/j.patter.2023.100758)  

**Our Paper:**  
[Provide full citation for your paper here]  

---
This repository serves as an adaptation of Reticula for mouse data, maintaining the foundational methodology while extending its application to new biological contexts.
