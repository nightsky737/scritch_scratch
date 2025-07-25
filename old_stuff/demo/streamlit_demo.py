'''
Gonna do a demo w/ streamlit

- Let you create a homemade model and overfit to it? Or use it for backprop on simple fxns? 
- Let you train jax and perhaps display metrics?
- let you train the cnn.

'''

import demo.streamlit_demo as st

st.write('Hello')
if st.button('click me'):
    if 'total' not in st.session_state:
        st.session_state.total=0 # or st.session_state['total'] = 0 -> kind of like uh the flask memory thing.
    st.session_state.total += 1
    st.write(st.session_state.total)