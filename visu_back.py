from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import polars as pl
import numpy as np
import boto3
from io import BytesIO
import math
from scipy import stats
from scipy.signal import find_peaks
import pandas as pd  # used only for Excel files conversion

app = FastAPI()

# Allow CORS (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# S3 configuration – update these values as needed.
S3_BUCKET = ""
S3_REGION = ""
s3_client = boto3.client("s3", region_name=S3_REGION)

# Global variable to hold our dataset (as a Polars DataFrame)
uploaded_df = None

def upload_to_s3(fileobj, key: str):
    s3_client.upload_fileobj(fileobj, S3_BUCKET, key)

def download_from_s3(key: str) -> BytesIO:
    fileobj = BytesIO()
    s3_client.download_fileobj(S3_BUCKET, key, fileobj)
    fileobj.seek(0)
    return fileobj

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a dataset, calculate basic categorical statistics,
    and provide available columns for cross-tabulation.
    """
    global uploaded_df

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    try:
        # Upload file to S3
        s3_key = f"uploads/{file.filename}"
        file_content = await file.read()
        upload_to_s3(BytesIO(file_content), s3_key)

        # Create a BytesIO for processing
        file_io = BytesIO(file_content)
        # Load the dataset using Polars
        if file.filename.endswith(".csv"):
            uploaded_df = pl.read_csv(file_io)
        elif file.filename.endswith(".xlsx"):
            # Polars does not natively support Excel so we use pandas then convert
            df_pd = pd.read_excel(file_io)
            uploaded_df = pl.from_pandas(df_pd)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Replace common missing value strings with None for text columns
        missing_vals = {"?", "N/A", "NA", "null", "Null", "--"}
        for col in uploaded_df.columns:
            if uploaded_df[col].dtype == pl.Utf8:
                uploaded_df = uploaded_df.with_column(
                    pl.col(col).apply(lambda x: None if x in missing_vals else x).alias(col)
                )

        # Identify categorical and numerical columns
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

        return JSONResponse(content={
            "categorical_statistics": categorical_statistics,
            "categorical_columns": categorical_cols,
            "columns": categorical_cols + numerical_cols,
            "filename": file.filename
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/normalize")
async def normalize(request: Request):
    """
    Applies normalization techniques to selected columns.
    """
    global uploaded_df

    if uploaded_df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON request")

    method = data.get('method')
    columns = data.get('columns')

    if not method or not columns:
        raise HTTPException(status_code=400, detail="Missing required fields")

    try:
        df = uploaded_df.clone()

        for col in columns:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in dataset")
            if df[col].dtype not in [pl.Int64, pl.Float64]:
                raise HTTPException(status_code=400, detail=f"Column '{col}' is not numeric and cannot be normalized")

        if method == "Z-Score Standardization":
            for col in columns:
                mean = df.select(pl.col(col).mean()).item()
                std = df.select(pl.col(col).std()).item()
                df = df.with_column(((pl.col(col) - mean) / std).alias(col))
        elif method == "Decimal Scaling":
            for col in columns:
                max_abs = df.select(pl.col(col).abs().max()).item()
                if max_abs != 0:
                    factor = 10 ** len(str(int(max_abs)))
                    df = df.with_column((pl.col(col) / factor).alias(col))
                else:
                    df = df.with_column(pl.lit(0).alias(col))
        elif method == "Min-Max Scaling":
            for col in columns:
                min_val = df.select(pl.col(col).min()).item()
                max_val = df.select(pl.col(col).max()).item()
                if math.isclose(max_val, min_val):
                    df = df.with_column(pl.lit(0).alias(col))
                else:
                    df = df.with_column(((pl.col(col) - min_val) / (max_val - min_val)).alias(col))
        elif method == "Exponential Normalization":
            for col in columns:
                df = df.with_column((pl.col(col).apply(np.exp)).alias(col))
        elif method == "Log Normalization":
            for col in columns:
                df = df.with_column((pl.col(col).apply(lambda x: np.log1p(x))).alias(col))
        else:
            raise HTTPException(status_code=400, detail="Invalid normalization method")

        uploaded_df = df
        normalized_data = df.select(columns).head(5).to_dicts()
        return JSONResponse(content={
            "message": f"Normalization completed using {method}",
            "normalized_data": normalized_data
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/handle-missing-data")
async def handle_missing_data(request: Request):
    """
    Endpoint to handle missing data using mean, median, mode or removal.
    """
    global uploaded_df
    if uploaded_df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON request")

    method = data.get("method")
    column = data.get("column", None)

    try:
        df = uploaded_df.clone()

        if method in ["mean", "median"]:
            if column:
                if df[column].dtype not in [pl.Int64, pl.Float64]:
                    raise HTTPException(status_code=400, detail=f"Cannot apply {method} imputation to non-numerical column: {column}")
                if method == "mean":
                    mean_val = df.select(pl.col(column).mean()).item()
                    df = df.with_column(pl.when(pl.col(column).is_null()).then(mean_val).otherwise(pl.col(column)).alias(column))
                else:
                    median_val = df.select(pl.col(column).median()).item()
                    df = df.with_column(pl.when(pl.col(column).is_null()).then(median_val).otherwise(pl.col(column)).alias(column))
            else:
                num_cols = [col for col in df.columns if df[col].dtype in [pl.Int64, pl.Float64]]
                for col in num_cols:
                    if method == "mean":
                        mean_val = df.select(pl.col(col).mean()).item()
                        df = df.with_column(pl.when(pl.col(col).is_null()).then(mean_val).otherwise(pl.col(col)).alias(col))
                    else:
                        median_val = df.select(pl.col(col).median()).item()
                        df = df.with_column(pl.when(pl.col(col).is_null()).then(median_val).otherwise(pl.col(col)).alias(col))
        elif method == "mode":
            if column:
                mode_val = df.select(pl.col(column).mode()).to_series()[0]
                df = df.with_column(pl.when(pl.col(column).is_null()).then(mode_val).otherwise(pl.col(column)).alias(column))
            else:
                for col in df.columns:
                    mode_val = df.select(pl.col(col).mode()).to_series()[0]
                    df = df.with_column(pl.when(pl.col(col).is_null()).then(mode_val).otherwise(pl.col(col)).alias(col))
        elif method == "remove":
            if column:
                df = df.filter(pl.col(column).is_not_null())
            else:
                df = df.drop_nulls()
        else:
            raise HTTPException(status_code=400, detail="Invalid method")

        uploaded_df = df
        missing_counts = {col: int(df.filter(pl.col(col).is_null()).height) 
                          for col in df.columns if df.filter(pl.col(col).is_null()).height > 0}
        return JSONResponse(content={
            "message": f"Missing data handled using {method} method.",
            "remaining_missing": missing_counts,
            "affected_columns": [column] if column else df.columns
        })
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
                    df = df.with_column(((pl.col(col) - mean) / std).alias(col))
                elif method == "Min-Max Scaling":
                    min_val = df.select(pl.col(col).min()).item()
                    max_val = df.select(pl.col(col).max()).item()
                    if math.isclose(max_val, min_val):
                        df = df.with_column(pl.lit(0).alias(col))
                    else:
                        df = df.with_column(((pl.col(col) - min_val) / (max_val - min_val)).alias(col))
                # Additional methods can be added here.
        buffer = BytesIO()
        df.write_csv(buffer)
        buffer.seek(0)

        # Optionally upload the normalized file to S3
        normalized_key = "normalized/normalized_dataset.csv"
        s3_client.upload_fileobj(buffer, S3_BUCKET, normalized_key)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=normalized_dataset.csv"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # For local development run: uvicorn main:app --reload
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
