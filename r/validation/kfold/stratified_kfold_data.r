#!/usr/bin/env Rscript

IN_DIR <- "/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/input/"
OUT_DIR <- "/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/output/"

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("Input directory:", IN_DIR, "\n")
cat("Output directory:", OUT_DIR, "\n")
cat("Output directory exists:", dir.exists(OUT_DIR), "\n")
cat("Output directory writable:", file.access(OUT_DIR, 2) == 0, "\n")

library(SummarizedExperiment)
library(DESeq2)
library(dplyr)
library(caret)

num_iterations <- 10

cat("Loading complete data...\n")
load(paste0(IN_DIR, "rse_gene_complete.Rdata"))
cat("Data loaded successfully.\n")

for (iter in 1:num_iterations) {
  
  cat(paste0("Processing iteration ", iter, " of ", num_iterations, "...\n"))
  
  iter_out_dir <- file.path(OUT_DIR, paste0("iter_", iter))
  dir.create(iter_out_dir, showWarnings = FALSE, recursive = TRUE)
  
  ensembl2rxns.df <- read.table(paste0(IN_DIR, "Ensembl2ReactomeReactions.txt"), sep="\t")
  
  tissue_data <- colData(final_result_filtered)
  
  tissue_counts <- table(tissue_data$Major_tissue)
  cat("Tissue counts:\n")
  print(tissue_counts)
  
  tissues_to_remove <- names(tissue_counts[tissue_counts <= 20])
  
  tissues_to_remove <- c(tissues_to_remove, 'Spleen')
  
  if (length(tissues_to_remove) > 0) {
    cat("Removing tissues with <=20 samples and Spleen:", paste(tissues_to_remove, collapse=", "), "\n")
    filtered_data <- final_result_filtered[, !(tissue_data$Major_tissue %in% tissues_to_remove)]
  } else {
    filtered_data <- final_result_filtered
  }
  
  row_totals <- rowSums(assays(filtered_data)$raw_counts)
  filtered_data <- filtered_data[row_totals > 0, ]
  
  gtex_tissues <- colData(filtered_data)$Major_tissue
  
  set.seed(iter * 100)  
  
  tissue_indices <- list()
  train_indices <- c()
  
  for (tissue in unique(gtex_tissues)) {
    tissue_idx <- which(gtex_tissues == tissue)
    tissue_indices[[tissue]] <- tissue_idx
    
    n_train <- floor(length(tissue_idx) * 0.8)
    
    train_idx <- sample(tissue_idx, n_train)
    train_indices <- c(train_indices, train_idx)
  }
  
  val_indices <- setdiff(1:length(gtex_tissues), train_indices)
  
  train_data <- filtered_data[, train_indices]
  val_data <- filtered_data[, val_indices]
  
  train_tissues <- colData(train_data)$Major_tissue
  val_tissues <- colData(val_data)$Major_tissue
  
  tryCatch({
    saveRDS(train_tissues, file=paste0(iter_out_dir, "/gtex_tissue_detail_vec_train_", iter, ".Rds"))
    saveRDS(val_tissues, file=paste0(iter_out_dir, "/gtex_tissue_detail_vec_val_", iter, ".Rds"))
    
    saveRDS(colData(train_data)$Row.names, file=paste0(iter_out_dir, "/gtex_sample_detail_vec_train_", iter, ".Rds"))
    saveRDS(colData(val_data)$Row.names, file=paste0(iter_out_dir, "/gtex_sample_detail_vec_val_", iter, ".Rds"))
    
    cat("Saved tissue and sample information successfully.\n")
  }, error = function(e) {
    cat("Error saving tissue or sample information:", conditionMessage(e), "\n")
    if (grepl("cannot open the connection", conditionMessage(e))) {
      cat("Directory check:", dir.exists(iter_out_dir), "\n")
      cat("Directory permissions:", file.access(iter_out_dir, 2), "\n")
      cat("Available disk space:\n")
      system("df -h")
    }
  })
  
  train_sample_ids <- colData(train_data)$external_id
  val_sample_ids <- colData(val_data)$external_id
  
  train_gtex.df <- train_data %>% SummarizedExperiment::assay() %>% as.data.frame()
  val_gtex.df <- val_data %>% SummarizedExperiment::assay() %>% as.data.frame()
  
  colnames(train_gtex.df) <- train_sample_ids
  colnames(val_gtex.df) <- val_sample_ids
  
  ensembl_wo_ids_train <- gsub("\\.[0-9]+", "", rownames(train_gtex.df))
  ensembl_wo_ids_val <- gsub("\\.[0-9]+", "", rownames(val_gtex.df))
  
  rownames(train_gtex.df) <- ensembl_wo_ids_train
  rownames(val_gtex.df) <- ensembl_wo_ids_val
  
  reactome_ensembl_ids_train <- intersect(ensembl2rxns.df$V1, ensembl_wo_ids_train)
  reactome_ensembl_ids_val <- intersect(ensembl2rxns.df$V1, ensembl_wo_ids_val)
  
  tryCatch({
    saveRDS(reactome_ensembl_ids_train, file=paste0(iter_out_dir, "/reactome_ensembl_ids_train_", iter, ".Rds"))
    saveRDS(reactome_ensembl_ids_val, file=paste0(iter_out_dir, "/reactome_ensembl_ids_val_", iter, ".Rds"))
    cat("Saved Reactome Ensembl IDs successfully.\n")
  }, error = function(e) {
    cat("Error saving Reactome Ensembl IDs:", conditionMessage(e), "\n")
  })
  
  train_gtex.df <- train_gtex.df[reactome_ensembl_ids_train, ]
  val_gtex.df <- val_gtex.df[reactome_ensembl_ids_val, ]
  
  tryCatch({
    saveRDS(train_gtex.df, file=paste0(iter_out_dir, "/gtex_df_train_", iter, ".Rds"))
    saveRDS(val_gtex.df, file=paste0(iter_out_dir, "/gtex_df_val_", iter, ".Rds"))
    cat("Saved filtered data successfully.\n")
  }, error = function(e) {
    cat("Error saving filtered data:", conditionMessage(e), "\n")
  })
  
  scale.factor.train <- (.Machine$integer.max - 1) / max(train_gtex.df)
  scale.factor.val <- (.Machine$integer.max - 1) / max(val_gtex.df)
  
  train_gtex.df <- round(train_gtex.df * scale.factor.train)
  val_gtex.df <- round(val_gtex.df * scale.factor.val)
  
  train_gtex.df <- train_gtex.df + 1
  val_gtex.df <- val_gtex.df + 1
  
  train_dds <- DESeq2::DESeqDataSetFromMatrix(
    countData = as.matrix(train_gtex.df),
    colData = data.frame(
      Sample = colnames(train_gtex.df),
      Tissue = train_tissues
    ),
    design = ~ Tissue
  )
  
  val_dds <- DESeq2::DESeqDataSetFromMatrix(
    countData = as.matrix(val_gtex.df),
    colData = data.frame(
      Sample = colnames(val_gtex.df),
      Tissue = val_tissues
    ),
    design = ~ Tissue
  )
  
  cat("Performing variance stabilizing transformation for training data...\n")
  train_vst.counts <- DESeq2::vst(
    train_dds,
    blind = FALSE,
    fitType = "local"
  )
  
  cat("Performing variance stabilizing transformation for validation data...\n")
  val_vst.counts <- DESeq2::vst(
    val_dds,
    blind = FALSE,
    fitType = "local"
  )
  
  tryCatch({
    saveRDS(train_vst.counts, file=paste0(iter_out_dir, "/vst_counts_train_", iter, ".Rds"))
    saveRDS(val_vst.counts, file=paste0(iter_out_dir, "/vst_counts_val_", iter, ".Rds"))
    cat("Saved VST counts successfully.\n")
  }, error = function(e) {
    cat("Error saving VST counts:", conditionMessage(e), "\n")
  })
  
  train_rxn2ensembls.nls <- list()
  train_rxns_w_gtex_ensembls.df <- ensembl2rxns.df %>% dplyr::filter(V1 %in% reactome_ensembl_ids_train)
  
  train_rxns_w_gtex_ensembls.df$V1 <- as.character(train_rxns_w_gtex_ensembls.df$V1)
  train_rxns_w_gtex_ensembls.df$V2 <- as.character(train_rxns_w_gtex_ensembls.df$V2)
  
  for (i in 1:nrow(train_rxns_w_gtex_ensembls.df)) {
    ens_id <- train_rxns_w_gtex_ensembls.df$V1[i]
    rxn_id <- train_rxns_w_gtex_ensembls.df$V2[i]
    ensembl_list_for_rxn_id <- train_rxn2ensembls.nls[[rxn_id]]
    
    if (is.null(ensembl_list_for_rxn_id)) {
      ensembl_list_for_rxn_id <- c(ens_id)
    }
    
    train_rxn2ensembls.nls[[rxn_id]] <- c(ensembl_list_for_rxn_id, ens_id) %>% unique()
  }
  
  val_rxn2ensembls.nls <- list()
  val_rxns_w_gtex_ensembls.df <- ensembl2rxns.df %>% dplyr::filter(V1 %in% reactome_ensembl_ids_val)
  
  val_rxns_w_gtex_ensembls.df$V1 <- as.character(val_rxns_w_gtex_ensembls.df$V1)
  val_rxns_w_gtex_ensembls.df$V2 <- as.character(val_rxns_w_gtex_ensembls.df$V2)
  
  for (i in 1:nrow(val_rxns_w_gtex_ensembls.df)) {
    ens_id <- val_rxns_w_gtex_ensembls.df$V1[i]
    rxn_id <- val_rxns_w_gtex_ensembls.df$V2[i]
    ensembl_list_for_rxn_id <- val_rxn2ensembls.nls[[rxn_id]]
    
    if (is.null(ensembl_list_for_rxn_id)) {
      ensembl_list_for_rxn_id <- c(ens_id)
    }
    
    val_rxn2ensembls.nls[[rxn_id]] <- c(ensembl_list_for_rxn_id, ens_id) %>% unique()
  }
  
  tryCatch({
    saveRDS(train_rxn2ensembls.nls, file=paste0(iter_out_dir, "/rxn2ensembls_nls_train_", iter, ".Rds"))
    saveRDS(val_rxn2ensembls.nls, file=paste0(iter_out_dir, "/rxn2ensembls_nls_val_", iter, ".Rds"))
    cat("Saved train and validation rxn2ensembls mappings successfully.\n")
  }, error = function(e) {
    cat("Error saving rxn2ensembls mappings:", conditionMessage(e), "\n")
  })
  
  train_vst.count.mtx <- train_vst.counts %>% SummarizedExperiment::assay() %>% as.data.frame()
  val_vst.count.mtx <- val_vst.counts %>% SummarizedExperiment::assay() %>% as.data.frame()
  
  tryCatch({
    saveRDS(train_vst.count.mtx, file=paste0(iter_out_dir, "/vst_count_mtx_train_", iter, ".Rds"))
    saveRDS(val_vst.count.mtx, file=paste0(iter_out_dir, "/vst_count_mtx_val_", iter, ".Rds"))
    cat("Saved count matrices successfully.\n")
  }, error = function(e) {
    cat("Error saving count matrices:", conditionMessage(e), "\n")
  })
  
  rxns <- train_rxn2ensembls.nls %>% names()
  val_rxns <- val_rxn2ensembls.nls %>% names()
  
  rxn_pca.nls <- list()
  full_rxn_pca_results.nls <- list()
  rxn_id_2_result_file_idx.nls <- list()

  val_rxn_pca.nls <- list()
  val_full_rxn_pca_results.nls <- list()
  val_rxn_id_2_result_file_idx.nls <- list()
  
  n_rxns <- length(rxns)
  result_idx <- 1

  val_n_rxns <- length(val_rxns)
  val_result_idx <- 1
  
  cat("Performing PCA for each reaction...\n")
  for (rxn_id_idx in seq(1:n_rxns)) {
    rxn_id <- rxns[rxn_id_idx]
    
    ensembl_ids <- train_rxn2ensembls.nls[[rxn_id]]
    
    if (!all(ensembl_ids %in% rownames(train_vst.count.mtx))) {
      ensembl_ids <- ensembl_ids[ensembl_ids %in% rownames(train_vst.count.mtx)]
      if (length(ensembl_ids) == 0) {
        next
      }
    }
    
    rxn_pca <- prcomp(t(train_vst.count.mtx[ensembl_ids, ]), scale. = TRUE)
    
    full_rxn_pca_results.nls[[rxn_id]] <- rxn_pca
    rxn_id_2_result_file_idx.nls[[rxn_id]] <- result_idx
    
    rxn_pca.nls[[rxn_id]] <- rxn_pca$x[, 1]
    
    if (rxn_id_idx %% 100 == 0) {
      cat(paste0("Processed ", rxn_id_idx, " of ", n_rxns, " reactions (", 
                round((rxn_id_idx + 1)/n_rxns, digits = 3) * 100, "%)...\n"))
    }
    
    if (rxn_id_idx %% 1000 == 0) {
      cat(paste0("Storing PCA objects containing reactions ", rxn_id_idx-1000, 
                "-", rxn_id_idx, " of ", n_rxns, " reactions (", 
                round((rxn_id_idx + 1)/n_rxns, digits = 3) * 100, "%)...\n"))
      
      tryCatch({
        saveRDS(full_rxn_pca_results.nls, 
                paste0(iter_out_dir, "/full_rxn_pca_results_nls", rxn_id_idx-1000, 
                      "-", rxn_id_idx, "_train_", iter, ".Rds"))
        cat("Saved intermediate PCA results successfully.\n")
      }, error = function(e) {
        cat("Error saving intermediate PCA results:", conditionMessage(e), "\n")
      })
      
      full_rxn_pca_results.nls <- list()
      gc()
      result_idx <- result_idx + 1
    }
  }

  for (rxn_id_idx in seq(1:val_n_rxns)) {
    rxn_id <- val_rxns[rxn_id_idx]
    
    ensembl_ids <- val_rxn2ensembls.nls[[rxn_id]]
    
    if (!all(ensembl_ids %in% rownames(val_vst.count.mtx))) {
      ensembl_ids <- ensembl_ids[ensembl_ids %in% rownames(val_vst.count.mtx)]
      if (length(ensembl_ids) == 0) {
        next
      }
    }
    
    rxn_pca <- prcomp(t(val_vst.count.mtx[ensembl_ids, ]), scale. = TRUE)
    
    val_full_rxn_pca_results.nls[[rxn_id]] <- rxn_pca
    val_rxn_id_2_result_file_idx.nls[[rxn_id]] <- val_result_idx
    
    val_rxn_pca.nls[[rxn_id]] <- rxn_pca$x[, 1]
    
    if (rxn_id_idx %% 100 == 0) {
      cat(paste0("Processed ", rxn_id_idx, " of ", val_n_rxns, " reactions (", 
                round((rxn_id_idx + 1)/val_n_rxns, digits = 3) * 100, "%)...\n"))
    }
    
    if (rxn_id_idx %% 1000 == 0) {
      cat(paste0("Storing PCA objects containing reactions ", rxn_id_idx-1000, 
                "-", rxn_id_idx, " of ", val_n_rxns, " reactions (", 
                round((rxn_id_idx + 1)/val_n_rxns, digits = 3) * 100, "%)...\n"))
      
      tryCatch({
        saveRDS(val_full_rxn_pca_results.nls, 
                paste0(iter_out_dir, "/val_full_rxn_pca_results_nls", rxn_id_idx-1000, 
                      "-", rxn_id_idx, "_train_", iter, ".Rds"))
        cat("Saved intermediate PCA results successfully.\n")
      }, error = function(e) {
        cat("Error saving intermediate PCA results:", conditionMessage(e), "\n")
      })
      
      val_full_rxn_pca_results.nls <- list()
      gc()
      val_result_idx <- val_result_idx + 1
    }
  }
  
  tryCatch({
    saveRDS(rxn_id_2_result_file_idx.nls, paste0(iter_out_dir, "/rxn_id_2_result_file_idx_nls_train_", iter, ".Rds"))
    saveRDS(full_rxn_pca_results.nls, paste0(iter_out_dir, "/full_rxn_pca_results_nls_train_", iter, ".Rds"))
    saveRDS(rxn_pca.nls, paste0(iter_out_dir, "/rxn_pca_nls_train_", iter, ".Rds"))
    cat("Saved final PCA results successfully.\n")
  }, error = function(e) {
    cat("Error saving final PCA results:", conditionMessage(e), "\n")
  })

  tryCatch({
    saveRDS(val_rxn_id_2_result_file_idx.nls, paste0(iter_out_dir, "/val_rxn_id_2_result_file_idx_nls_train_", iter, ".Rds"))
    saveRDS(val_full_rxn_pca_results.nls, paste0(iter_out_dir, "/val_full_rxn_pca_results_nls_train_", iter, ".Rds"))
    saveRDS(val_rxn_pca.nls, paste0(iter_out_dir, "/val_rxn_pca_nls_train_", iter, ".Rds"))
    cat("Saved final PCA results successfully.\n")
  }, error = function(e) {
    cat("Error saving final PCA results:", conditionMessage(e), "\n")
  })
  
  E <- tryCatch({
    read.table(paste0(IN_DIR, "ReactionNetwork_Rel.txt"))
  }, error = function(e) {
    cat("Error reading reaction network:", conditionMessage(e), "\n")
    return(NULL)
  })

  val_E <- tryCatch({
    read.table(paste0(IN_DIR, "ReactionNetwork_Rel.txt"))
  }, error = function(e) {
    cat("Error reading reaction network:", conditionMessage(e), "\n")
    return(NULL)
  })
  

  
  rxn2nodeLabel.nls <- list()
  nodeLabel2rxn.nls <- list()

  val_rxn2nodeLabel.nls <- list()
  val_nodeLabel2rxn.nls <- list()
  
  for (i in 1:length(rxn_pca.nls)) {
    rxn2nodeLabel.nls[[names(rxn_pca.nls)[i]]] <- i
    nodeLabel2rxn.nls[[i]] <- names(rxn_pca.nls)[i]
  }

  for (i in 1:length(val_rxn_pca.nls)) {
    val_rxn2nodeLabel.nls[[names(val_rxn_pca.nls)[i]]] <- i
    val_nodeLabel2rxn.nls[[i]] <- names(val_rxn_pca.nls)[i]
  }
  
  E <- E %>%
    dplyr::filter(V1 %in% names(rxn2nodeLabel.nls)) %>%
    dplyr::filter(V3 %in% names(rxn2nodeLabel.nls)) %>%
    dplyr::select(V1, V3)
  
  val_E <- val_E %>%
    dplyr::filter(V1 %in% names(val_rxn2nodeLabel.nls)) %>%
    dplyr::filter(V3 %in% names(val_rxn2nodeLabel.nls)) %>%
    dplyr::select(V1, V3)
  
  tryCatch({
    write.table(E, file=paste0(IN_DIR, "edgeLabels_train_", iter, ".csv"),
                row.names = FALSE, col.names = FALSE)
    cat("Saved edge labels successfully.\n")
  }, error = function(e) {
    cat("Error saving edge labels:", conditionMessage(e), "\n")
  })

  tryCatch({
    write.table(val_E, file=paste0(IN_DIR, "val_edgeLabels_train_", iter, ".csv"),
                row.names = FALSE, col.names = FALSE)
    cat("Saved edge labels successfully.\n")
  }, error = function(e) {
    cat("Error saving edge labels:", conditionMessage(e), "\n")
  })
  
  node1 <- numeric()
  node2 <- numeric()
  
  for (i in 1:nrow(E)) {
    node1 <- c(node1, rxn2nodeLabel.nls[[as.character(E$V1[i])]])
    node2 <- c(node2, rxn2nodeLabel.nls[[as.character(E$V3[i])]])
  }
  
  z <- unlist(rxn2nodeLabel.nls)
  y <- unlist(nodeLabel2rxn.nls)

  val_node1 <- numeric()
  val_node2 <- numeric()
  
  for (i in 1:nrow(val_E)) {
    val_node1 <- c(val_node1, val_rxn2nodeLabel.nls[[as.character(val_E$V1[i])]])
    val_node2 <- c(val_node2, val_rxn2nodeLabel.nls[[as.character(val_E$V3[i])]])
  }
  
  val_z <- unlist(val_rxn2nodeLabel.nls)
  val_y <- unlist(val_nodeLabel2rxn.nls)
  
  tryCatch({
    write.table(z, file=paste0(IN_DIR, "rxn2nodeLabel_nls_train_", iter, ".csv"),
                row.names = TRUE, col.names = FALSE)
    write.table(y, file=paste0(IN_DIR, "nodeLabel2rxn_nls_train_", iter, ".csv"),
                row.names = TRUE, col.names = FALSE)
    cat("Saved node mappings successfully.\n")
  }, error = function(e) {
    cat("Error saving node mappings:", conditionMessage(e), "\n")
  })

  tryCatch({
    write.table(val_z, file=paste0(IN_DIR, "val_rxn2nodeLabel_nls_train_", iter, ".csv"),
                row.names = TRUE, col.names = FALSE)
    write.table(val_y, file=paste0(IN_DIR, "val_nodeLabel2rxn_nls_train_", iter, ".csv"),
                row.names = TRUE, col.names = FALSE)
    cat("Saved node mappings successfully.\n")
  }, error = function(e) {
    cat("Error saving node mappings:", conditionMessage(e), "\n")
  })
  
  E <- data.frame(node1 = node1, node2 = node2)
  
  X <- as.data.frame(rxn_pca.nls)
  Y <- as.data.frame(train_tissues)

  val_E <- data.frame(node1 = val_node1, node2 = val_node2)
  
  val_X <- as.data.frame(val_rxn_pca.nls)
  val_Y <- as.data.frame(val_tissues)
  
  tryCatch({
    write.table(E, file=paste0(IN_DIR, "edges_train_", iter, ".txt"),
                row.names = FALSE, col.names = FALSE)
    write.table(X, file=paste0(IN_DIR, "node_features_train_", iter, ".txt"),
                row.names = FALSE, col.names = FALSE)
    write.table(Y, file=paste0(IN_DIR, "graph_targets_train_", iter, ".txt"),
                row.names = FALSE, col.names = FALSE)
    write.table(X, file=paste0(IN_DIR, "node_features2_train_", iter, ".txt"),
                row.names = TRUE, col.names = TRUE)
    cat("Saved GNN training data successfully.\n")
  }, error = function(e) {
    cat("Error saving GNN training data:", conditionMessage(e), "\n")
  })
  
  tryCatch({
    write.table(val_E, file=paste0(IN_DIR, "edges_val_", iter, ".txt"),
                row.names = FALSE, col.names = FALSE)
    write.table(val_X, file=paste0(IN_DIR, "node_features_val_", iter, ".txt"),
                row.names = FALSE, col.names = FALSE)
    write.table(val_Y, file=paste0(IN_DIR, "graph_targets_val_", iter, ".txt"),
                row.names = FALSE, col.names = FALSE)
    cat("Saved GNN training data successfully.\n")
  }, error = function(e) {
    cat("Error saving GNN training data:", conditionMessage(e), "\n")
  })
  
  cat(paste0("Completed iteration ", iter, " of ", num_iterations, "\n"))
}

cat("All iterations completed successfully.\n")