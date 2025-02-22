# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import polars as pl
# import pandas as pd  # fallback for Excel and narrative generation
# import io
# import uuid
# import json
# import boto3
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from datetime import datetime

# app = FastAPI()

# # Allow CORS for all origins (adjust as needed)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize boto3 S3 client (make sure AWS credentials are configured)
# s3_client = boto3.client("s3")
# BUCKET_NAME = "your-s3-bucket-name"  # Replace with your bucket name

# # In-memory storage for project metadata
# projects = {}

# def upload_file_to_s3(file_bytes: bytes, s3_key: str):
#     try:
#         s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=file_bytes)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"S3 upload error: {str(e)}")

# def download_file_from_s3(s3_key: str) -> io.BytesIO:
#     try:
#         file_buffer = io.BytesIO()
#         s3_client.download_fileobj(BUCKET_NAME, s3_key, file_buffer)
#         file_buffer.seek(0)
#         return file_buffer
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"S3 download error: {str(e)}")

# def read_dataset(file_buffer: io.BytesIO, filename: str) -> pl.DataFrame:
#     filename = filename.lower()
#     if filename.endswith(".csv"):
#         df = pl.read_csv(file_buffer)
#     elif filename.endswith((".xls", ".xlsx")):
#         # Fallback to Pandas for Excel then convert to Polars
#         df_pd = pd.read_excel(file_buffer)
#         df = pl.from_pandas(df_pd)
#     elif filename.endswith(".json"):
#         data = json.load(file_buffer)
#         df = pl.DataFrame(data)
#     else:
#         raise HTTPException(status_code=400, detail="Unsupported file format")
#     return df

# @app.get("/")
# def index():
#     return {"message": "Welcome to the Data API", "projects": list(projects.values())}

# @app.get("/dashboard")
# def dashboard():
#     return {"projects": list(projects.values())}

# @app.post("/upload")
# async def upload_file(
#     file: UploadFile = File(...),
#     project_name: str = Form("Untitled Project"),
#     project_description: str = Form("")
# ):
#     filename = file.filename
#     file_format = filename.lower()
#     try:
#         # Read file bytes once
#         contents = await file.read()
#         # Create a unique S3 key for the file (you may include folder structure)
#         s3_key = f"datasets/{uuid.uuid4()}_{filename}"
#         # Upload the raw file to S3
#         upload_file_to_s3(contents, s3_key)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

#     project_id = str(uuid.uuid4())
#     projects[project_id] = {
#         "id": project_id,
#         "name": project_name,
#         "description": project_description,
#         "s3_key": s3_key,  # store reference to the S3 object
#         "file_format": file_format,
#         "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     }
#     return {"project_id": project_id, "project_name": project_name, "status": "uploaded"}

# def load_project_dataset(project: dict) -> pl.DataFrame:
#     s3_key = project.get("s3_key")
#     file_format = project.get("file_format", "")
#     file_buffer = download_file_from_s3(s3_key)
#     df = read_dataset(file_buffer, file_format)
#     return df

# @app.get("/project/{project_id}")
# def get_project(project_id: str):
#     if project_id not in projects:
#         raise HTTPException(status_code=404, detail="Project not found")
#     return projects[project_id]

# @app.post("/analyze")
# def analyze_data(project_id: str):
#     if project_id not in projects:
#         raise HTTPException(status_code=404, detail="Project not found")
#     try:
#         df = load_project_dataset(projects[project_id])
#         # Select numeric columns based on Polars schema
#         numeric_cols = [
#             col for col, dtype in df.schema.items() 
#             if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
#                          pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
#                          pl.Float32, pl.Float64)
#         ]
#         numeric_df = df.select(numeric_cols)
#         if numeric_df.is_empty():
#             raise HTTPException(status_code=400, detail="No numeric data available for analysis")

#         # Correlation analysis
#         correlation = numeric_df.corr().to_dict()

#         # Determine number of PCA components (min(2, columns, rows))
#         n_components = min(2, numeric_df.width, numeric_df.height)
#         if n_components < 2:
#             pca_result = {
#                 "x": numeric_df[:, 0].to_list() if not numeric_df.is_empty() else [],
#                 "y": [0] * numeric_df.height if not numeric_df.is_empty() else []
#             }
#             scaled_data = None
#         else:
#             data_np = numeric_df.to_numpy()
#             scaler = StandardScaler()
#             scaled_data = scaler.fit_transform(data_np)
#             pca = PCA(n_components=n_components)
#             pca_result_array = pca.fit_transform(scaled_data)
#             pca_result = {
#                 "x": pca_result_array[:, 0].tolist(),
#                 "y": pca_result_array[:, 1].tolist() if n_components > 1 else [0] * len(pca_result_array)
#             }

#         # Perform KMeans clustering if there are at least 3 rows
#         if scaled_data is not None and numeric_df.height >= 3:
#             n_clusters = min(3, numeric_df.height)
#             kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#             kmeans_result = kmeans.fit_predict(scaled_data).tolist()
#         else:
#             kmeans_result = [0] * numeric_df.height

#         analysis = {
#             "correlation": correlation,
#             "pca": pca_result,
#             "kmeans": kmeans_result,
#         }
#         return analysis

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

# @app.post("/export")
# def export_data(project_id: str, export_format: str = "csv"):
#     if project_id not in projects:
#         raise HTTPException(status_code=404, detail="Project not found")
#     try:
#         df = load_project_dataset(projects[project_id])
#         project_name = projects[project_id]["name"]
#         export_format = export_format.lower()
#         if export_format == "csv":
#             data = df.write_csv(string=True)
#             filename = f"{project_name}.csv"
#             file_content = data  # text-based export
#         elif export_format == "json":
#             data = df.to_json()
#             filename = f"{project_name}.json"
#             file_content = data
#         elif export_format == "excel":
#             # Fallback to Pandas for Excel export
#             df_pd = df.to_pandas()
#             output = io.BytesIO()
#             df_pd.to_excel(output, index=False)
#             file_content = output.getvalue().hex()  # return hex string for binary data
#             filename = f"{project_name}.xlsx"
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported export format")
#         return {"filename": filename, "file_content": file_content}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

# @app.post("/generate-story")
# def generate_story(project_id: str):
#     if project_id not in projects:
#         raise HTTPException(status_code=404, detail="Project not found")
#     try:
#         # For narrative generation, we use Pandas for convenience
#         df = load_project_dataset(projects[project_id]).to_pandas()
#         narrative = generate_narrative(df)
#         insights = generate_insights(df)
#         visualizations = suggest_visualizations(df)
#         return {
#             "narrative": narrative,
#             "insights": insights,
#             "visualizations": visualizations,
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Story generation error: {str(e)}")

# def generate_narrative(df):
#     narrative = []
#     narrative.append("<div class='mb-6'>")
#     narrative.append("<h3 class='text-2xl font-bold mb-4'>📊 Your Data Story</h3>")
#     narrative.append(f"<p class='text-lg'>Analyzed {len(df)} records with fascinating insights.</p>")
#     narrative.append("</div>")
#     return "".join(narrative)

# def generate_insights(df):
#     insights = []
#     numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
#     for col in numeric_cols:
#         mean = df[col].mean()
#         median = df[col].median()
#         if abs(mean - median) > df[col].std():
#             insights.append(f"The distribution of {col} is skewed.")
#     return " ".join(insights) if insights else "No significant insights."

# def suggest_visualizations(df):
#     numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     if len(numeric_cols) >= 2:
#         return [{
#             "title": "Scatter Plot",
#             "description": f"Visualize relationship between {numeric_cols[0]} and {numeric_cols[1]}",
#             "type": "scatter",
#             "config": {"x": numeric_cols[0], "y": numeric_cols[1]}
#         }]
#     return []

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



