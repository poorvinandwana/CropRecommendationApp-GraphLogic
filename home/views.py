from django.shortcuts import render
from graphrag.retrieval.retrieval_pipeline import recommend_crop

MOISTURE_MAP = {
    "high": 70,
    "moderate-high": 60,
    "moderate": 50,
    "moderate-low": 40,
    "low": 30,
}

SALINITY_MAP = {
    "low": 1,
    "moderate-low": 2,
    "moderate": 3,
    "moderate-high": 4,
    "high": 5,
}

def recommend_view(request):
    if request.method == "POST":
        soil_data = {
            "N": float(request.POST["nitrogen"]),
            "P": float(request.POST["phosphorus"]),
            "K": float(request.POST["potassium"]),
            "T": float(request.POST["temperature"]),
            "pH": float(request.POST["ph"]),
            "M": MOISTURE_MAP[request.POST["moisture"]],
            "salinity": SALINITY_MAP[request.POST["salinity"]],
            "soil_type": request.POST["soil_type"].capitalize(),
        }

        result = recommend_crop(soil_data)

        return render(request, "result.html", {
            "result": result,
            "soil": soil_data
        })

    return render(request, "index.html")

