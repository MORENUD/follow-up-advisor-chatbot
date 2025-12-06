# vistualize.py
from graph import app

try:
    png_data = app.get_graph().draw_mermaid_png()
    output_file = "my_medical_graph.png"
    with open(output_file, "wb") as f:
        f.write(png_data)
    
    print(f"✅ Graph successfully saved to: {output_file}")
    print("You can view the file now!")

except Exception as e:
    print(f"❌ Error generating graph image: {e}")
    print("\n--- Mermaid Code (You can copy/paste this into mermaid.live) ---")
    print(app.get_graph().draw_mermaid())