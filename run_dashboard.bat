@echo off
title SkyFlow - Air Traffic Congestion Forecasting Console
echo =====================================================================
echo  SkyFlow: Real-Time Air Traffic Congestion Forecasting & DSS Console
echo =====================================================================
echo.
echo Launching Streamlit Dashboard on http://localhost:8501 ...
echo.
cd /d "%~dp0"
streamlit run dashboard/app.py
pause
