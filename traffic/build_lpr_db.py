import sqlite3
import random
import string

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]

def generate_license_plate():
    return ''.join(random.choices(string.ascii_uppercase, k=3)) + ''.join(random.choices(string.digits, k=4))

def generate_random_state():
    return random.choice(US_STATES)

def build():
    conn = sqlite3.connect("us_license_plates.db")
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS license_plates")
    cursor.execute("""
        CREATE TABLE license_plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT UNIQUE,
            state TEXT
        )
    """)

    plates = [(generate_license_plate(), generate_random_state()) for _ in range(1000)]
    cursor.executemany("INSERT INTO license_plates (plate, state) VALUES (?, ?)", plates)
    conn.commit()

    cursor.execute("SELECT * FROM license_plates LIMIT 10")
    print(cursor.fetchall())

    conn.close()

def db_query(plate):
    conn = sqlite3.connect("us_license_plates.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT plate, state FROM license_plates WHERE plate = ?", (plate,))
    result = cursor.fetchone()
    
    if result:
        conn.close()
        return f"Plate found: {result[0]}, state: {result[1]}"
    else:
        return "Plate not found"
        cursor.execute("INSERT INTO license_plates (plate, state) VALUES (?, ?)", (plate, "TX"))
        conn.commit()
        conn.close()
        return "Plate not found, inserted into database"

# [(1, 'NGD0206', 'AK'), (2, 'WSG5446', 'NV'), (3, 'HQD4328', 'UT'), (4, 'OAD8938', 'OR'), (5, 'CZO9092', 'WY'), (6, 'CBY2399', 'MA'), (7, 'KDV0867', 'LA'), (8, 'IEQ6573', 'IN'), (9, 'XNV1901', 'SD'), (10, 'BVC2997', 'OH')]
if __name__ == "__main__":
    build()

    pass            