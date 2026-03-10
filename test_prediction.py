from predict_helper import predict_news

sample_news = """
Government announces new economic reforms after an official cabinet meeting in parliament.
"""

result = predict_news(sample_news)

print("Prediction Result")
print("-" * 40)
print("Prediction:", result["prediction"])
print("Raw Class:", result["raw_prediction"])
print("Score:", result["score"])
print("Interpretation:", result["interpretation"])