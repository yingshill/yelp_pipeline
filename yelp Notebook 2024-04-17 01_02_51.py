# Databricks notebook source
# Create a Spark session

from pyspark.sql import SparkSession

# Create SparkSession object
spark = SparkSession.builder.appName("YelpDataAnalysis").getOrCreate()

# COMMAND ----------

# Load the JSON data into a DataFrame
business_df = spark.read.json("/Volumes/yelp_project1_analysis/yelp/yelp_data/yelp_academic_dataset_business.json")
user_df = spark.read.json("/Volumes/yelp_project1_analysis/yelp/yelp_data/yelp_academic_dataset_user.json")
review_df = spark.read.json("/Volumes/yelp_project1_analysis/yelp/yelp_data/yelp_academic_dataset_review.json")
tip_df = spark.read.json("/Volumes/yelp_project1_analysis/yelp/yelp_data/yelp_academic_dataset_tip.json")
checkin_df = spark.read.json("/Volumes/yelp_project1_analysis/yelp/yelp_data/yelp_academic_dataset_checkin.json")

# COMMAND ----------

def show_df(map_list):
    """
    Show DataFrame
    """
    for df in map_list:
        df.show()
        display(df)


# COMMAND ----------

# Flatten business data
from pyspark.sql.functions import col, explode_outer
from pyspark.sql.types import StructType, ArrayType, StructField
from pyspark.sql import DataFrame

# A generic function to flattens a table with nested structures
## Identify the nested fields and properly handle them by either exploding arrays or extracting nested fields.
### Step 1: Analyze the Schema
business_df.printSchema()

### Step 2: Define the flattening function
def flatten_schema(schema, prefix=""):
    """
    Flatten the schema and return a list of column expressions.
    Handles nested structs and arrays recursively.
    """
    fields = []
    for field in schema.fields:
        name = prefix + field.name
        dtype = field.dataType
        if isinstance(dtype, StructType):
            # Recursively flatten structs
            fields += flatten_schema(dtype, prefix=name + ".")
        elif isinstance(dtype, ArrayType) and isinstance(dtype.elementType, StructType):
            # Explode arrays of structs and flatten further
            fields.append(explode_outer(col(name)).alias(name))
            fields += flatten_schema(dtype.elementType, prefix=name + ".")
        else:
            # For basic data types or arrays of basic data types
            fields.append(col(name).alias(name.replace(".", "_")))
    return fields

def flatten_df(df):
    """
    Flatten the DataFrame by resolving all nested structs and arrays.
    """
    # Get flat column expressions from the schema
    flat_cols = flatten_schema(df.schema)
    
    # Select these columns to get a flat DataFrame
    df_flat = df.select(*flat_cols)
    
    # Handle exploded arrays by grouping and pivoting (if necessary) or any additional logic
    # This part can be customized based on specific requirements
    return df_flat


# COMMAND ----------

# Get flattened DataFrames
flat_business_df = flatten_df(business_df)
flat_user_df = flatten_df(user_df)
flat_checkin_df = flatten_df(checkin_df)
flat_review_df = flatten_df(review_df)
flat_tip_df = flatten_df(tip_df)

flat_df_list = [flat_business_df, flat_user_df, flat_checkin_df, flat_review_df, flat_tip_df]
show_df(flat_df_list)

# flattened_business_df.write.format("json").mode("overwrite").save("/Volumes/yelp_project1_analysis/yelp/yelp_data/flattened_business.json")

# COMMAND ----------

def rename_cols(df, rename_map):
    """
    Renames cols in a DataFrame based on a dictionary mapping of old names to new names.

    Parameters:
    - df(DataFrame): The Spark DataFrame whose cols are to be renamed.
    - rename_map(dict): A dictionary mapping original cols names to new col names.

    Returns:
    - DataFrame: A DataFrame with cols renamed as specified in rename_map.
    """
    for old_name, new_name in rename_map.items():
        df = df.withColumnRenamed(old_name, new_name)
    return df

business_rename_map = {
    "name": "business_name",
    "review_count": "business_review_count",
    "stars": "business_stars",
    "categories": "business_categories",
    "address": "business_address"
    
}

user_rename_map = {
    "average_stars": "user_average_stars",
    "name": "user_name",
    "review_count": "user_review_count",
    "useful": "user_review_count",
    "funny":  "user_funny_votes",
    "cool": "user_cool_votes",
    "yelping_since": "user_yelping_since",
    "friends": "user_friends",
    "elite": "user_elite",
    "fans": "user_fans"         
}

checkin_rename_map = {
    "date": "checkin_date"
}

review_rename_map = {
    "stars": "review_stars",
    "useful": "review_useful_votes",
    "funny": "review_funny_votes",
    "cool": "review_cool_votes",
    "date": "review_date",
    "text": "review_text"
}

tip_rename_map = {
    "text": "tip_text",
    "date": "tip_date",
    "compliment_count": "tip_compliment_count"
}

renamed_business_df = rename_cols(flat_business_df, business_rename_map)
renamed_user_df = rename_cols(flat_user_df, user_rename_map)
renamed_checkin_df = rename_cols(flat_checkin_df, checkin_rename_map)
renamed_review_df = rename_cols(flat_review_df, review_rename_map)
renamed_tip_df = rename_cols(flat_tip_df, tip_rename_map)

renamed_business_df.show()
renamed_user_df.show()
renamed_checkin_df.show()
renamed_review_df.show()
renamed_tip_df.show()


# COMMAND ----------

# Handle duplicates
renamed_business_df = renamed_business_df.repartition("business_id")
renamed_user_df = renamed_user_df.repartition("user_id")
renamed_review_df = renamed_review_df.repartition("review_id")

business_df_no_duplicates = renamed_business_df.dropDuplicates()
user_df_no_duplicates = renamed_user_df.dropDuplicates()
checkin_df_no_duplicates = renamed_checkin_df.dropDuplicates()
review_df_no_duplicates = renamed_review_df.dropDuplicates()
tip_df_no_duplicates = renamed_tip_df.dropDuplicates()

no_duplicates_df_list = [business_df_no_duplicates, user_df_no_duplicates, checkin_df_no_duplicates, review_df_no_duplicates, tip_df_no_duplicates]
show_df(no_duplicates_df_list)


# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


def remove_u_prefix(s):
    """
    Remove the 'u' prefix from a string if it exists.
    Example: u'Some string' => 'Some string'
    """
    if s is None:
        return None
    elif s.startswith(u"u'") and s.endswith("'") and len(s) > 2:
        return s[2:-1]
    elif s.startswith('u"') and s.endswith('"') and len(s) > 2:
        return s[2:-1]
    return s

# Register the function as a UDF
remove_u_prefix_udf = udf(remove_u_prefix, StringType())

def clean_string_cols(df):
    """
    Apply the 'remove_u_prefix_udf' to all string cols in the DataFrame
    """

    # Iterate over DataFrame cols and apply the UDF to string cols
    for column, dtype in df.dtypes:
        if dtype == 'string':
            df = df.withColumn(column, remove_u_prefix_udf(col(column)))
    return df

cleaned_business_df = clean_string_cols(business_df_no_duplicates)
cleaned_user_df = clean_string_cols(renamed_user_df)
cleaned_checkin_df = clean_string_cols(renamed_checkin_df)
cleaned_review_df = clean_string_cols(renamed_review_df)
cleaned_tip_df = clean_string_cols(renamed_tip_df)


# COMMAND ----------

# Reorder fields in business DataFrame
business_cols = cleaned_business_df.columns
print(business_cols)
ordered_cols = ['business_name', 'business_id', 'business_address', 'business_categories', 'city', 'latitude', 'longitude', 'postal_code', 'business_review_count', 'business_stars', 'state', 'hours_Friday', 'hours_Monday', 'hours_Saturday', 'hours_Sunday', 'hours_Thursday', 'hours_Tuesday', 'hours_Wednesday', 'is_open', 'attributes_AcceptsInsurance', 'attributes_AgesAllowed', 'attributes_Alcohol', 'attributes_Ambience', 'attributes_BYOB', 'attributes_BYOBCorkage', 'attributes_BestNights', 'attributes_BikeParking', 'attributes_BusinessAcceptsBitcoin', 'attributes_BusinessAcceptsCreditCards', 'attributes_BusinessParking', 'attributes_ByAppointmentOnly', 'attributes_Caters', 'attributes_CoatCheck', 'attributes_Corkage', 'attributes_DietaryRestrictions', 'attributes_DogsAllowed', 'attributes_DriveThru', 'attributes_GoodForDancing', 'attributes_GoodForKids', 'attributes_GoodForMeal', 'attributes_HairSpecializesIn', 'attributes_HappyHour', 'attributes_HasTV', 'attributes_Music', 'attributes_NoiseLevel', 'attributes_Open24Hours', 'attributes_OutdoorSeating', 'attributes_RestaurantsAttire', 'attributes_RestaurantsCounterService', 'attributes_RestaurantsDelivery', 'attributes_RestaurantsGoodForGroups', 'attributes_RestaurantsPriceRange2', 'attributes_RestaurantsReservations', 'attributes_RestaurantsTableService', 'attributes_RestaurantsTakeOut', 'attributes_Smoking', 'attributes_WheelchairAccessible', 'attributes_WiFi']
reordered_business_df = cleaned_business_df.select(*ordered_cols)
display(reordered_business_df)

# COMMAND ----------

# Join the DataFrames

# Join review to business
joined_df = reordered_business_df.join(cleaned_review_df, "business_id", "left_outer")

# Join check-in data
joined_df = joined_df.join(cleaned_checkin_df, "business_id", "left_outer")

# Join tip data
joined_df = joined_df.join(cleaned_tip_df, ["business_id", "user_id"], "left_outer")

# Join user data
joined_df = joined_df.join(cleaned_user_df, "user_id", "left_outer")

# Dropping unnecessary columns
final_df = joined_df.drop("user_elite")

final_df.show()

# COMMAND ----------

spark.stop()
