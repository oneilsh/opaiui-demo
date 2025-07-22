from pydantic_ai import Agent
from pydantic_ai.tools import RunContext
from pydantic_ai.mcp import MCPServerStdio

import streamlit as st
from opaiui.app import AgentConfig, AppConfig, AgentState, call_render_func, serve


import dotenv
dotenv.load_dotenv(override = True)

###############
## Agent Definition
###############

semantic_scholar_mcp = MCPServerStdio(
    command = 'poetry',
    args = ["run", "python", "arxiv_mcp.py"],
)

scholar_agent = Agent('openai:gpt-4o', toolsets = [semantic_scholar_mcp])


################
## Deps and Tools
################

class Library():
    def __init__(self):
        self.state = AgentState()
        self.state.library = []

    def add(self, article: str):
        """Save an article to the library."""
        self.state.library.append(article)

    def as_markdown(self) -> str:
        if not self.state.library:
            return "None"
        return "\n".join(f"- {entry}" for entry in self.state.library)

@scholar_agent.tool
async def add_to_library(ctx: RunContext[Library], article: str) -> str:
    """Add a given article to the library."""
    ctx.deps.add(article)
    return f"Article added. Current library size: {len(ctx.deps.state.library)}"


################
## Streamlit Sidebar
################

# will be given the deps object 
async def scholar_sidebar(deps):
    """Render the agent's sidebar in Streamlit."""
    st.markdown("### Library")
    st.markdown(deps.as_markdown())

    def clear_library():
        """Clear the library."""
        deps.state.library = []
    
    if st.button("Clear Library"):
        clear_library()
        st.rerun()


################
## Agent Config
################

# We configure UI elements and set dependencies for agents, as a dictionary
# mapping agent names to AgentConfig instances.

agent_configs = {
    "arXiv Bot": AgentConfig(agent = scholar_agent,
                             deps = Library(),
                             sidebar_func = scholar_sidebar,
                             greeting= "Hello! What should we learn about today?",
                             agent_avatar= "📖")}



## Global app configuration configures page title, icon, default sidebar state, default function call visibility, etc.


import pandas

async def render_df(df: pandas.DataFrame):
    """Render a DataFrame in Streamlit."""
    st.dataframe(df, use_container_width=True)

async def show_warning(message: str):
    """Display a warning message in Streamlit."""
    st.warning(message)

render_funcs = [render_df, show_warning]



app_config = AppConfig(sidebar_collapsed= False,
                       page_icon= "📖",
                       page_title= "arXiv Bot",
                       rendering_functions= render_funcs,)



@scholar_agent.tool
async def show_library(ctx: RunContext[Library]) -> str:
    """Displays the current library to the user as a dataframe when executed."""
    if not ctx.deps.state.library:
        await call_render_func("show_warning", {"message": "Library is empty."}, before_agent_response = True)
        return "Library is empty. A warning has been displayed to the user prior to this response."
    
    df = pandas.DataFrame(ctx.deps.state.library, columns=["Articles"])
    await call_render_func("render_df", {"df": df})
    return "Library will be displayed as a DataFrame *below* your response in the chat. You may refer to it, but do not repeat the library contents in your response."


#################
## Run the app
#################

serve(app_config, agent_configs)

