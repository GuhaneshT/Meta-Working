from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import polars as pl
import numpy as np
import os
from io import BytesIO
import math
from scipy import stats
from scipy.signal import find_peaks
import pandas as pd  # used for Excel conversion

app = FastAPI()

# Allow CORS (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local directories for file storage
UPLOAD_FOLDER = "uploads"
NORMALIZED_FOLDER = "normalized_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(NORMALIZED_FOLDER, exist_ok=True)

# Global variable to hold our dataset (as a Polars DataFrame)
uploaded_df = None

# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     """
#     Endpoint to upload a dataset, calculate basic categorical statistics,
#     and provide available columns for cross-tabulation.
#     """
#     global uploaded_df

#     if not file.filename:
#         raise HTTPException(status_code=400, detail="No file selected")

#     try:
#         file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         contents = await file.read()
#         with open(file_path, "wb") as f:
#             f.write(contents)

#         file_io = BytesIO(contents)
#         if file.filename.endswith(".csv"):
#             uploaded_df = pl.read_csv(file_io)
#         elif file.filename.endswith(".xlsx"):
#             # Use pandas to read Excel, then convert to Polars
#             df_pd = pd.read_excel(file_io)
#             uploaded_df = pl.from_pandas(df_pd)
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported file format")

#         # Replace common missing value strings with None for string columns
#         missing_vals = {"?", "N/A", "NA", "null", "Null", "--"}
#         for col in uploaded_df.columns:
#             if uploaded_df[col].dtype == pl.Utf8:
#                 uploaded_df = uploaded_df.with_column(
#                     pl.col(col).apply(lambda x: None if x in missing_vals else x).alias(col)
#                 )

#         # Identify categorical and numerical columns
#         categorical_cols = [col for col, dt in zip(uploaded_df.columns, uploaded_df.dtypes) if dt == pl.Utf8]
#         numerical_cols = [col for col, dt in zip(uploaded_df.columns, uploaded_df.dtypes) if dt in [pl.Int64, pl.Float64]]

#         # Calculate categorical statistics
#         categorical_statistics = {}
#         for col in categorical_cols:
#             series = uploaded_df[col]
#             vc_df = series.value_counts()
#             counts = {row[0]: row[1] for row in vc_df.to_numpy()}
#             total_non_null = series.drop_nulls().n_unique()
#             mode_list = series.mode().to_list()
#             mode_val = mode_list[0] if mode_list else None
#             total_count = sum(counts.values()) if counts else 0
#             rel_freq = {k: round((v / total_count) * 100, 2) if total_count > 0 else 0 for k, v in counts.items()}
#             categorical_statistics[col] = {
#                 "unique_values": total_non_null,
#                 "most_frequent": mode_val,
#                 "frequency_counts": counts,
#                 "relative_frequency": rel_freq,
#             }

#         return JSONResponse(content={
#             "categorical_statistics": categorical_statistics,
#             "categorical_columns": categorical_cols,
#             "columns": categorical_cols + numerical_cols,
#             "filename": file.filename
#         })

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global uploaded_df

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    try:
        print("Received file:", file.filename)
        contents = await file.read()
        print("File size:", len(contents))
        
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        file_io = BytesIO(contents)
        if file.filename.endswith(".csv"):
            uploaded_df = pl.read_csv(file_io, encoding="utf8")
        elif file.filename.endswith(".xlsx"):
            import pandas as pd
            df_pd = pd.read_excel(file_io)
            uploaded_df = pl.from_pandas(df_pd)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        print("DataFrame loaded. Columns:", uploaded_df.columns)
        
        missing_vals = {"?", "N/A", "NA", "null", "Null", "--"}
        for col in uploaded_df.columns:
            if uploaded_df[col].dtype == pl.Utf8:
        # Convert the column to a list, apply the lambda, and then convert back to a Series.
                new_series = pl.Series(col, [None if isinstance(x, str) and x in missing_vals else x 
                                     for x in uploaded_df[col].to_list()])
                uploaded_df = uploaded_df.with_columns(new_series)


        categorical_cols = [col for col, dt in zip(uploaded_df.columns, uploaded_df.dtypes) if dt == pl.Utf8]
        numerical_cols = [col for col, dt in zip(uploaded_df.columns, uploaded_df.dtypes) if dt in [pl.Int64, pl.Float64]]
        
        # Calculate categorical statistics
        categorical_statistics = {}
        for col in categorical_cols:
            series = uploaded_df[col]
            vc_df = series.value_counts()
            counts = {row[0]: row[1] for row in vc_df.to_numpy()}
            total_non_null = series.drop_nulls().n_unique()
            mode_list = series.mode().to_list()
            mode_val = mode_list[0] if mode_list else None
            total_count = sum(counts.values()) if counts else 0
            rel_freq = {k: round((v / total_count) * 100, 2) if total_count > 0 else 0 for k, v in counts.items()}
            categorical_statistics[col] = {
                "unique_values": total_non_null,
                "most_frequent": mode_val,
                "frequency_counts": counts,
                "relative_frequency": rel_freq,
            }
        
        preview_df = uploaded_df.head(5)
        headers = preview_df.columns
        rows = [list(row.values()) for row in preview_df.to_dicts()]
        print("Returning preview with headers:", headers)
        
        return JSONResponse(content={
            "categorical_statistics": categorical_statistics,
            "categorical_columns": categorical_cols,
            "columns": categorical_cols + numerical_cols,
            "filename": file.filename,
            "headers": headers,
            "rows": rows
        })
    except Exception as e:
        print("Error occurred:", e)
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/missingvalue")
# async def missingvalue(request : Request)
# @app.post("/normalize")
# async def normalize(request: Request):
#     """
#     Applies normalization techniques to selected columns.
#     """
#     global uploaded_df

#     if uploaded_df is None:
#         raise HTTPException(status_code=400, detail="No dataset uploaded")

#     try:
#         data = await request.json()
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid JSON request")

#     method = data.get('method')
#     columns = data.get('columns')

#     if not method or not columns:
#         raise HTTPException(status_code=400, detail="Missing required fields")

#     try:
#         df = uploaded_df.clone()

#         # Validate columns
#         for col in columns:
#             if col not in df.columns:
#                 raise HTTPException(status_code=400, detail=f"Column '{col}' not found in dataset")
#             if df[col].dtype not in [pl.Int64, pl.Float64]:
#                 raise HTTPException(status_code=400, detail=f"Column '{col}' is not numeric and cannot be normalized")

#         # Apply normalization methods
#         if method == "Z-Score Standardization":
#             for col in columns:
#                 mean = df.select(pl.col(col).mean()).item()
#                 std = df.select(pl.col(col).std()).item()
#                 df = df.with_columns(((pl.col(col) - mean) / std).alias(col))
#         elif method == "Decimal Scaling":
#             for col in columns:
#                 max_abs = df.select(pl.col(col).abs().max()).item()
#                 if max_abs != 0:
#                     factor = 10 ** len(str(int(max_abs)))
#                     df = df.with_columns((pl.col(col) / factor).alias(col))
#                 else:
#                     df = df.with_columns(pl.lit(0).alias(col))
#         elif method == "Min-Max Scaling":
#             for col in columns:
#                 min_val = df.select(pl.col(col).min()).item()
#                 max_val = df.select(pl.col(col).max()).item()
#                 if math.isclose(max_val, min_val):
#                     df = df.with_columns(pl.lit(0).alias(col))
#                 else:
#                     df = df.with_columns(((pl.col(col) - min_val) / (max_val - min_val)).alias(col))
#         elif method == "Exponential Normalization":
#             for col in columns:
#                 df = df.with_columns((pl.col(col).map(np.exp)).alias(col))
#         elif method == "Log Normalization":
#             for col in columns:
#                 df = df.with_columns((pl.col(col).map(lambda x: np.log1p(x))).alias(col))
#         else:
#             raise HTTPException(status_code=400, detail="Invalid normalization method")

#         uploaded_df = df
#         normalized_data = df.select(columns).head(5).to_dicts()
#         return JSONResponse(content={
#             "message": f"Normalization completed using {method}",
#             "normalized_data": normalized_data
#         })

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



# Global variable holding our uploaded DataFrame



@app.post("/preprocess")
async def preprocess(request: Request):
    """
    Endpoint to preprocess data based on the provided options.
    
    Expected JSON payload:
    {
      "handleMissing": "drop" | "mean" | "median" | "mode",
      "filterColumn": "optional column name",
      "filterValue": "optional filter value",
      "transformations": ["normalize", "standardize", "log", "square_root"]
    }
    """
    global uploaded_df
    if uploaded_df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON request.")

    handle_missing = data.get("handleMissing")
    filter_column = data.get("filterColumn")
    filter_value = data.get("filterValue")
    transformations = data.get("transformations", [])

    df = uploaded_df.clone()  # Work on a copy

    # ----- 1. Handle Missing Values -----
    if handle_missing not in {"drop", "mean", "median", "mode"}:
        raise HTTPException(status_code=400, detail="Invalid handleMissing method.")

    if handle_missing in {"mean", "median"}:
        # If a specific column is provided, apply to that column; else, apply to all numeric columns.
        if filter_column:
            if df[filter_column].dtype not in [pl.Int64, pl.Float64]:
                raise HTTPException(status_code=400, detail=f"Column '{filter_column}' is not numeric for {handle_missing} imputation.")
            replacement = (df.select(pl.col(filter_column).mean()).item() 
                           if handle_missing == "mean" 
                           else df.select(pl.col(filter_column).median()).item())
            new_series = pl.Series(filter_column, [
                replacement if (x is None) else x for x in df[filter_column].to_list()
            ])
            df = df.with_columns([new_series])
        else:
            num_cols = [col for col, dt in zip(df.columns, df.dtypes) if dt in [pl.Int64, pl.Float64]]
            for col in num_cols:
                replacement = (df.select(pl.col(col).mean()).item() 
                               if handle_missing == "mean" 
                               else df.select(pl.col(col).median()).item())
                new_series = pl.Series(col, [
                    replacement if (x is None) else x for x in df[col].to_list()
                ])
                df = df.with_columns([new_series])
    elif handle_missing == "mode":
        if filter_column:
            counts = df.select(pl.col(filter_column)).to_series().value_counts().to_dict(as_series=False)
            replacement = max(counts, key=lambda k: counts[k]) if counts else None
            new_series = pl.Series(filter_column, [
                replacement if (x is None) else x for x in df[filter_column].to_list()
            ])
            df = df.with_columns([new_series])
        else:
            for col in df.columns:
                counts = df.select(pl.col(col)).to_series().value_counts().to_dict(as_series=False)
                replacement = max(counts, key=lambda k: counts[k]) if counts else None
                new_series = pl.Series(col, [
                    replacement if (x is None) else x for x in df[col].to_list()
                ])
                df = df.with_columns([new_series])
    elif handle_missing == "drop":
        if filter_column:
            df = df.filter(pl.col(filter_column).is_not_null())
        else:
            df = df.drop_nulls()

    # ----- 2. Data Transformations -----
    # Supported transformations: "normalize", "standardize", "log", "square_root"
    num_cols = [col for col, dt in zip(df.columns, df.dtypes) if dt in [pl.Int64, pl.Float64]]
    for transform in transformations:
        if transform == "normalize":
            for col in num_cols:
                min_val = df.select(pl.col(col).min()).item()
                max_val = df.select(pl.col(col).max()).item()
                if math.isclose(max_val, min_val):
                    new_series = pl.Series(col, [0] * df.shape[0])
                else:
                    new_series = pl.Series(col, [
                        (x - min_val) / (max_val - min_val) if x is not None else None
                        for x in df[col].to_list()
                    ])
                df = df.with_columns([new_series])
        elif transform == "standardize":
            for col in num_cols:
                mean_val = df.select(pl.col(col).mean()).item()
                std_val = df.select(pl.col(col).std()).item()
                if std_val == 0 or std_val is None:
                    new_series = pl.Series(col, [0] * df.shape[0])
                else:
                    new_series = pl.Series(col, [
                        (x - mean_val) / std_val if x is not None else None
                        for x in df[col].to_list()
                    ])
                df = df.with_columns([new_series])
        elif transform == "log":
            for col in num_cols:
                new_series = pl.Series(col, [
                    np.log1p(x) if (isinstance(x, (int, float)) and x >= 0) else x
                    for x in df[col].to_list()
                ])
                df = df.with_columns([new_series])
        elif transform == "square_root":
            for col in num_cols:
                new_series = pl.Series(col, [
                    np.sqrt(x) if (isinstance(x, (int, float)) and x >= 0) else x
                    for x in df[col].to_list()
                ])
                df = df.with_columns([new_series])
        else:
            # Ignore unrecognized transformation
            pass

    # ----- 3. Filter Data -----
    # ----- 3. Filter Data -----
    if filter_column and filter_value is not None:
    # Cast the column to Utf8 if needed before filtering
        df = df.filter(pl.col(filter_column).cast(pl.Utf8).str.contains(filter_value, literal=True))


    # Update the global DataFrame
    uploaded_df = df

    missing_counts = {col: df.filter(pl.col(col).is_null()).height
                      for col in df.columns if df.filter(pl.col(col).is_null()).height > 0}

    return JSONResponse(content={
        "message": "Preprocessing complete.",
        "missing_counts": missing_counts,
        "columns": df.columns,
        "num_rows": df.shape[0]
    })

@app.get("/statistics")
async def dataset_statistics():
    """
    Endpoint to retrieve dataset statistics.
    """
    global uploaded_df

    if uploaded_df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a dataset first.")

    try:
        stats_dict = {
            "rows": uploaded_df.shape[0],
            "columns": len(uploaded_df.columns),
            "duplicates": uploaded_df.shape[0] - uploaded_df.unique().shape[0]
        }
        return JSONResponse(content=stats_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/get_column_metadata")
async def get_column_metadata():
    """
    Returns column metadata: names, data types, and null counts.
    """
    global uploaded_df
    if uploaded_df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded")

    try:
        metadata = []
        for col in uploaded_df.columns:
            null_count = uploaded_df.filter(pl.col(col).is_null()).height
            dtype = "Numeric" if uploaded_df[col].dtype in [pl.Int64, pl.Float64] else "Categorical"
            metadata.append({"columnName": col, "dataType": dtype, "nullCount": null_count})
        return JSONResponse(content={"column_metadata": metadata})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/covariance-correlation")
async def covariance_correlation(request: Request):
    """
    Endpoint to calculate covariance and correlation between two numerical columns.
    """
    global uploaded_df
    if uploaded_df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a dataset first.")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON request")

    column1 = data.get("column1")
    column2 = data.get("column2")
    if not column1 or not column2:
        raise HTTPException(status_code=400, detail="Both column1 and column2 are required.")

    try:
        if column1 not in uploaded_df.columns or column2 not in uploaded_df.columns:
            raise HTTPException(status_code=400, detail="Invalid columns selected.")
        if uploaded_df[column1].dtype not in [pl.Int64, pl.Float64] or uploaded_df[column2].dtype not in [pl.Int64, pl.Float64]:
            raise HTTPException(status_code=400, detail="Both columns must be numerical.")

        arr1 = np.array(uploaded_df[column1])
        arr2 = np.array(uploaded_df[column2])
        covariance_value = np.cov(arr1, arr2)[0, 1]
        correlation_value = np.corrcoef(arr1, arr2)[0, 1]
        scatter_data = [{"x": float(x), "y": float(y)} for x, y in zip(arr1, arr2)]
        return JSONResponse(content={
            "covariance": round(covariance_value, 2),
            "correlation": round(correlation_value, 2),
            "scatterData": scatter_data
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download")
async def download_normalized(request: Request):
    """
    Endpoint to download the normalized dataset.
    The dataset is written to an in-memory CSV buffer and returned.
    """
    global uploaded_df
    if uploaded_df is None:
        raise HTTPException(status_code=400, detail="No dataset available")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON request")

    method = data.get('method')
    columns_to_normalize = data.get('columns', [])
    try:
        df = uploaded_df.clone()
        for col in columns_to_normalize:
            if col in df.columns:
                if method == "Z-Score Standardization":
                    mean = df.select(pl.col(col).mean()).item()
                    std = df.select(pl.col(col).std()).item()
                    df = df.with_columns(((pl.col(col) - mean) / std).alias(col))
                elif method == "Min-Max Scaling":
                    min_val = df.select(pl.col(col).min()).item()
                    max_val = df.select(pl.col(col).max()).item()
                    if math.isclose(max_val, min_val):
                        df = df.with_columns(pl.lit(0).alias(col))
                    else:
                        df = df.with_columns(((pl.col(col) - min_val) / (max_val - min_val)).alias(col))
                # Additional methods can be added here.
        # Write CSV to a local file
        output_filename = "normalized_dataset.csv"
        output_path = os.path.join(NORMALIZED_FOLDER, output_filename)
        df.write_csv(output_path)
        # Return file as a streaming response
        def iterfile():
            with open(output_path, mode="rb") as file_like:
                yield from file_like

        return StreamingResponse(iterfile(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={output_filename}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/columns")
async def get_columns():
    """
    Endpoint to retrieve the columns of the uploaded dataset.
    """
    global uploaded_df
    if uploaded_df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a dataset first.")

    return JSONResponse(content=uploaded_df.columns)
from fastapi import Query
from typing import List
# @app.get("/visualize")
# async def visualize(
#     columns: List[str] = Query(..., description="List of columns to visualize"),
#     type: str = Query("bar", description="Type of chart: bar, line, pie, scatter")
# ):
#     """
#     Build chart data from the uploaded DataFrame.

#     The first column in the 'columns' parameter is used as labels,
#     and subsequent columns (if any) are used as datasets.
#     For a pie chart, only the first dataset is used.
#     """
#     global uploaded_df
#     if uploaded_df is None:
#         raise HTTPException(status_code=400, detail="No dataset uploaded.")

#     # Parse the comma-separated columns
#     selected_columns = [col.strip() for col in columns.split(",") if col.strip()]
#     if not selected_columns:
#         raise HTTPException(status_code=400, detail="No columns provided.")

#     # Ensure selected columns exist in the DataFrame
#     for col in selected_columns:
#         if col not in uploaded_df.columns:
#             raise HTTPException(status_code=400, detail=f"Column '{col}' not found in dataset.")

#     # Use the first column for labels.
#     labels = uploaded_df[selected_columns[0]].to_list()

#     datasets = []
#     # Use remaining columns as datasets
#     for col in selected_columns[1:]:
#         # Ensure the dataset column is numeric.
#         if uploaded_df[col].dtype not in [pl.Int64, pl.Float64]:
#             raise HTTPException(status_code=400, detail=f"Column '{col}' must be numeric for visualization.")
#         data_values = uploaded_df[col].to_list()
#         dataset = {
#             "label": col,
#             "data": data_values,
#             "backgroundColor": "rgba(75, 192, 192, 0.4)",
#             "borderColor": "rgba(75, 192, 192, 1)",
#             "borderWidth": 1
#         }
#         datasets.append(dataset)

#     # For pie charts, only use the first dataset.
#     if type.lower() == "pie" and datasets:
#         datasets = [datasets[0]]

#     chart_data = {
#         "labels": labels,
#         "datasets": datasets
#     }

#     return JSONResponse(content=chart_data)


@app.get("/visualize")
async def visualize(
    columns: List[str] = Query(..., description="List of columns to visualize"),
    type: str = Query("bar", description="Type of chart: bar, line, pie, scatter")
):
    """
    Build chart data from the uploaded DataFrame.

    Uses the first column in 'columns' as labels, and remaining columns as datasets.
    For pie charts, only the first dataset is returned.
    """
    global uploaded_df
    if uploaded_df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")

    # Ensure selected columns exist in the DataFrame
    for col in columns:
        if col not in uploaded_df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found in dataset.")

    # Use the first column as labels
    labels = uploaded_df[columns[0]].to_list()

    datasets = []
    # Use remaining columns as datasets
    for col in columns[1:]:
        if uploaded_df[col].dtype not in [pl.Int64, pl.Float64]:
            raise HTTPException(status_code=400, detail=f"Column '{col}' must be numeric for visualization.")
        data_values = uploaded_df[col].to_list()
        dataset = {
            "label": col,
            "data": data_values,
            "backgroundColor": "rgba(75, 192, 192, 0.4)",
            "borderColor": "rgba(75, 192, 192, 1)",
            "borderWidth": 1
        }
        datasets.append(dataset)

    # For pie charts, use only the first dataset.
    if type.lower() == "pie" and datasets:
        datasets = [datasets[0]]

    chart_data = {
        "labels": labels,
        "datasets": datasets
    }

    return JSONResponse(content=chart_data)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
