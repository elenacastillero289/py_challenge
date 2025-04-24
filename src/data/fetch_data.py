import requests 
import json

def get_data(seed: int, number_of_datapoints: int) -> list[dict[str, any]]:
    """
    Fetches animal data from the external data-service API.

    Args:
        seed (int): Seed to generate reproducible data.
        number_of_datapoints (int): Number of data points to fetch.

    Returns:
        list[dict[str, any]]: A list of data records returned from the API.
    """
    url = "http://localhost:8777/api/v1/animals/data"
    params = {
        "seed": seed,
        "number_of_datapoints": number_of_datapoints
    }
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }    

    response = requests.post(url, headers=headers, data=json.dumps(params))
    response.raise_for_status()

    return response.json()

def save_data_to_file(data: list[dict[str, any]], filepath: str) -> None:
    """
    Saves the fetched data to a JSON file.

    Args:
        data (list[dict[str, any]]): Data to be saved.
        filepath (str): Path to the output file.
    """

    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    seed = 42
    number_of_datapoints = 1000
    try:
        data = get_data(seed, number_of_datapoints)
        print(f"Retrieved {len(data)} datapoints.")
        save_data_to_file(data, "src/data/animals.json")
        print("Data saved successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")

  

