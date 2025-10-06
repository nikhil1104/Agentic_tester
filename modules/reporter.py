"""Simple Reporter (Phase 1)
- Generates an HTML + JSON summary for the run
""" 
import os, json
from jinja2 import Template

OUT = "reports"
os.makedirs(OUT, exist_ok=True)

class Reporter:
    def __init__(self):
        pass

    def create_reports(self, results: dict) -> dict:
        eid = results.get("execution_id")
        json_path = os.path.join(OUT, f"{eid}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # simple HTML
        html_template = """<html><body><h1>Run {{eid}}</h1>
        {% for suite, cases in results.test_results.items() %}
        <h2>{{suite}}</h2>
        {% for c in cases %}
          <h3>{{c.name}}</h3>
          <ul>
            {% for st in c.steps %}
              <li>{{st.step}} - {{st.status}}</li>
            {% endfor %}
          </ul>
        {% endfor %}
        {% endfor %}
        </body></html>"""
        tpl = Template(html_template)
        html = tpl.render(eid=eid, results=results)
        html_path = os.path.join(OUT, f"{eid}.html")
        with open(html_path, "w") as f:
            f.write(html)
        return {"json": json_path, "html": html_path}
