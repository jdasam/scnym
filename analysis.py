import pandas as pd
import numpy as np
from pathlib import Path

if __name__ == "__main__":
      csv_path = "geninus_prediction_scnym_ssl.csv"

      df = pd.read_csv(csv_path)
      gen_pred = df["pred_celltype_l2"].values
      sc_pred = df["celltype_pred_scnym"].values

      print(f'Number of same predictions are {np.sum(gen_pred==sc_pred)} among total {len(gen_pred)} cells, which is ratio of {np.sum(gen_pred==sc_pred)/len(gen_pred)}.')

      confusion_matrix = pd.crosstab(df['pred_celltype_l2'], df['celltype_pred_scnym'], rownames=['Geninus'], colnames=['scNym'])
      confusion_matrix['Total'] = confusion_matrix.sum(axis=1)
      confusion_matrix.loc["Total"] = confusion_matrix.sum()

      for key in confusion_matrix.keys():
            if key not in ["Total", "Recall"]:
                  confusion_matrix.at["Precision", key]= confusion_matrix[key][key] / confusion_matrix[key]["Total"]
      for key in confusion_matrix.keys():
            if key not in ["Total", "Recall", "Precision"]:
                  confusion_matrix.at[key, "Recall"]= confusion_matrix[key][key] / confusion_matrix["Total"][key]

      confusion_matrix.to_csv("confusion_matrix_geninus_scnym_ssl.csv")

      seurat_dir = Path("/home/svcapp/tbrain_x/SKT_data_prediction_by_Seurat")
      csv_files = seurat_dir.glob("*.csv")
      seurat_df = pd.concat([pd.read_csv(csv) for csv in csv_files])

      df = df.merge(seurat_df)

      sr_pred = df["predicted.celltype.l2"]
      print(f'Geninus-Seurat: Number of same predictions are {np.sum(gen_pred==sr_pred)} among total {len(gen_pred)} cells, which is ratio of {np.sum(gen_pred==sr_pred)/len(gen_pred)}.')
      print(f'scNym-Seurat: Number of same predictions are {np.sum(sc_pred==sr_pred)} among total {len(sc_pred)} cells, which is ratio of {np.sum(sc_pred==sr_pred)/len(sc_pred)}.')

      confusion_matrix = pd.crosstab(df['pred_celltype_l2'], df['predicted.celltype.l2'], rownames=['Geninus'], colnames=['Seurat'])
      # confusion_matrix.to_csv("confusion_matrix_geninus_seurat.csv")
      # confusion_matrix = pd.crosstab(df['celltype_pred_scnym'], df['predicted.celltype.l2'], rownames=['scNym'], colnames=['Seurat'])
      # confusion_matrix.to_csv("confusion_matrix_scnym_ssl_seurat.csv")

