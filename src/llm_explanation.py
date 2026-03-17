from openai import OpenAI

client = OpenAI()


def explain_feature_importance(feature_df):

    features_text = "\n".join(
        [f"{row['feature']}: {row['importance']:.3f}" for _, row in feature_df.iterrows()]
    )

    prompt = f"""
Explain why these features are important for electricity price forecasting:

{features_text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = response.choices[0].message.content

    except Exception:

        explanation = (
            "Lagged prices dominate because electricity prices show strong "
            "autocorrelation. Net load captures the balance between demand "
            "and renewable generation."
        )

    log_llm(prompt, explanation)

    return explanation


def interpret_trading_signal(forecast_price, market_price):

    prompt = f"""
Forecast price: {forecast_price:.2f} €/MWh
Market price: {market_price:.2f} €/MWh

Explain whether this suggests bullish or bearish pressure
in the day-ahead electricity market.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = response.choices[0].message.content

    except Exception:

        if forecast_price < market_price:
            explanation = "Forecast below market price suggests bearish pressure."
        else:
            explanation = "Forecast above market price suggests bullish pressure."

    log_llm(prompt, explanation)

    return explanation


def generate_market_report(forecast_price, market_price, signal):

    prompt = f"""
Write a short electricity market summary.

Forecast price: {forecast_price:.2f} €/MWh
Market price: {market_price:.2f} €/MWh
Trading signal: {signal}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        report = response.choices[0].message.content

    except Exception:

        report = (
            f"Forecast price is {forecast_price:.2f} €/MWh compared "
            f"to market price {market_price:.2f} €/MWh. "
            f"Trading signal: {signal}."
        )

    log_llm(prompt, report)

    return report


def log_llm(prompt, output):

    with open("qa_output/llm_log.txt", "a") as f:

        f.write("PROMPT:\n")
        f.write(prompt)

        f.write("\n\nOUTPUT:\n")
        f.write(output)

        f.write("\n\n----------------\n")