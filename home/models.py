from django.db import models


class Crop(models.Model):
    name = models.CharField(max_length=100, unique=True)

    # Macronutrient Requirements
    nitrogen_min = models.FloatField()
    nitrogen_max = models.FloatField()
    phosphorus_min = models.FloatField()
    phosphorus_max = models.FloatField()
    potassium_min = models.FloatField()
    potassium_max = models.FloatField()

    # Other Soil Requirements
    ph_min = models.FloatField()
    ph_max = models.FloatField()
    temperature_min = models.FloatField()
    temperature_max = models.FloatField()
    soil_type = models.CharField(
    max_length=15,
    choices=[
        ("clay", "Clay"),
        ("sandy", "Sandy"),
        ("loamy", "Loamy"),
        ("peaty", "Peaty"),
        ("saline", "Saline"),
        ("silty", "Silty"),
    ],
)


    # Moisture Content
    moisture_content = models.CharField(
        max_length=20,
        choices=[
            ("high", "High"),
            ("moderate-high", "Moderate-High"),
            ("moderate", "Moderate"),
            ("moderate-low", "Moderate-Low"),
            ("low", "Low"),
        ],
        default="moderate",
    )

    # Salinity (Can have multiple values)
    salinity = models.CharField(
        max_length=20,
        choices=[
            ("high", "High"),
            ("moderate-high", "Moderate-High"),
            ("moderate", "Moderate"),
            ("moderate-low", "Moderate-Low"),
            ("low", "Low"),
        ],
        default="moderate",
    )

    def __str__(self):
        return self.name
