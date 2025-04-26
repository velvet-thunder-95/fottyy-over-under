# Savable Filters Documentation

This document details how the **savable filter** feature works, covering both `filter storage.py` and `app.py`, with file paths, line numbers, and step-by-step flow.

---

## 1. filter_storage.py


```python
# load_saved_filters (lines 13–31)
13 def load_saved_filters():
14     """Load filters from Supabase"""
15     try:
17         response = supabase.table('saved_filters') \
18             .select('*') \
19             .order('created_at', desc=True) \
20             .execute()
22         if response.data:
23             return [{
24                 'id': f['id'],
25                 'name': f['name'],
26                 'leagues': f['leagues'],
27                 'confidence': f['confidence'],
28                 'created': f['created_at']
29             } for f in response.data]
30         return []
31     except Exception as e:
32         st.error(f"Error loading filters: {str(e)}")
33         return []
```
- **Purpose:** Fetch all saved filters, sorted newest first.
- **Returns:** List of dicts with keys `id`, `name`, `leagues`, `confidence`, `created`.

```python
# save_filter (lines 35–49)
35 def save_filter(name, leagues, confidence_levels):
36     """Save a new filter to Supabase"""
37     try:
38         data = {
39             'name': name,
40             'leagues': leagues,
41             'confidence': confidence_levels
42         }
44         response = supabase.table('saved_filters').insert(data).execute()
46         if response.data:
47             return load_saved_filters()  # return updated list
48         return []
49     except Exception as e:
50         st.error(f"Error saving filter: {str(e)}")
51         return []
```
- **Purpose:** Insert a new filter; on success, reloads all filters.

```python
# delete_filter (lines 53–64)
53 def delete_filter(filter_id):
54     """Delete a filter from Supabase"""
55     try:
56         supabase.table('saved_filters') \
57             .delete() \
58             .eq('id', filter_id) \
59             .execute()
61         return load_saved_filters()  # return updated list
62     except Exception as e:
63         st.error(f"Error deleting filter: {str(e)}")
64         return []
```
- **Purpose:** Remove a filter by `id`; returns refreshed list.

---

## 2. app.py


### 2.1 Local wrappers

```python
# save_filter wrapper (lines 2433–2445)
2433 def save_filter(name, leagues, confidence_levels):
2434     """Save filter to Supabase"""
2435     try:
2436         # payload built but not used here directly
2437         data = {...}
2442         return filter_storage.save_filter(name, leagues, confidence_levels)
2438     except Exception as e:
2439         logger.error(f"Error saving filter: {str(e)}")
2440         return []
```
- Delegates to `filter_storage.save_filter` and returns its result.

```python
# delete_filter wrapper (lines 2447–2453)
2447 def delete_filter(filter_id):
2448     """Delete filter from Supabase"""
2449     try:
2450         return filter_storage.delete_filter(filter_id)
2451     except Exception as e:
2452         logger.error(f"Error deleting filter: {str(e)}")
2453         return []
```

### 2.2 Initial load in UI

```python
# show_main_app (lines 2456–2459)
2457 if 'saved_filters' not in st.session_state:
2458     st.session_state.saved_filters = filter_storage.load_saved_filters()
```
- Populates `st.session_state.saved_filters` once at startup.

### 2.3 Filter UI section

```python
# UI block (lines 2549–2595)
2549 with st.container():
2554     filter_name = st.text_input("Save Filter", key="filter_name")
2556     if st.button("Save", ...):
2557         if filter_name:
2558             st.session_state.saved_filters = save_filter(
2559                 filter_name,
2560                 selected_leagues,
2561                 confidence_levels
2562             )
2563             st.success(...)
2565         else:
2566             st.error(...)
2568 if st.session_state.saved_filters:
2570     for idx, sf in enumerate(st.session_state.saved_filters):
2584         if st.button("Apply", key=f"apply_filter_{idx}"):
2585             st.session_state.selected_leagues = sf['leagues']
2586             st.session_state.confidence_levels = sf['confidence']
2587             st.rerun()
2590         if st.button("Delete", key=f"delete_filter_{idx}"):
2591             st.session_state.saved_filters = delete_filter(sf['id'])
2592             st.rerun()
```
- **Save**: user enters `filter_name`, clicks **Save** ➔ calls wrapper ➔ Supabase ➔ reload UI list.
- **Apply**: sets session state to saved values + reruns app.
- **Delete**: calls wrapper, updates UI list + reruns.

---

## 3. Usage Flow
1. **Startup:** load existing filters.
2. **Define filter:** choose leagues & confidence, name it, click **Save**.
3. **Persist:** new filter saved in Supabase; UI list refreshed.
4. **Apply/Delete:** interact with saved filters instantly.

---

### Quick Start
1. Ensure your Supabase project has a `saved_filters` table with columns:
   - `id` (UUID, primary key)
   - `name` (text)
   - `leagues` (text[])
   - `confidence` (text[])
   - `created_at` (timestamp with timezone, default now())
2. Export environment variables:
   ```bash
   export SUPABASE_URL="https://<project>.supabase.co"
   export SUPABASE_KEY="<YOUR_SUPABASE_KEY>"
   ```
3. Install dependencies:
   ```bash
   pip install supabase streamlit
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Configuration
- In `filter storage.py`, replace hard-coded keys:
  ```python
  SUPABASE_URL = os.getenv("SUPABASE_URL")
  SUPABASE_KEY = os.getenv("SUPABASE_KEY")
  ```
- Confirm `supabase.table('saved_filters')` uses the correct table name and columns.

### Detailed Usage
1. **Load Filters**: on app start, existing filters are fetched via `load_saved_filters()` into `st.session_state.saved_filters`.
2. **Define a Filter**:
   - Select leagues and confidence levels from the multiselects.
   - Enter a unique name in the **Save Filter** input.
   - Click **Save** to call `save_filter()`, which inserts into Supabase and reloads all filters.
3. **View Saved Filters**: below the save section, each saved filter appears with its name, leagues, and confidence.
4. **Apply a Filter**: click **Apply** next to a filter to set selections back and refresh the UI.
5. **Delete a Filter**: click **Delete** to call `delete_filter()`, remove from Supabase, and refresh the list.
6. **Session State**: selections persist in `st.session_state.selected_leagues` and `confidence_levels`.

### Code Locations
- `filter storage.py`: CRUD functions at lines **13–33**, **35–51**, **53–64**.
- `app.py`: wrappers at lines **2433–2445**, **2447–2453**, and UI integration at **2549–2595**.

---

**Note:** Secure `SUPABASE_KEY` via environment variable; ensure Supabase schema matches `{name: text, leagues: array, confidence: array, created_at: timestamp}`.
