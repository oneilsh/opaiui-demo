from pydantic_ai import Agent
from pydantic_ai.tools import RunContext
from pydantic_ai.mcp import MCPServerStdio
import sys

import streamlit as st
from opaiui.app import AgentConfig, AppConfig, AgentState, call_render_func, serve


import dotenv
dotenv.load_dotenv(override = True)

###############
## Agent Definition
###############

# This agent uses the MCP server defined in arxiv_mcp.py to search for papers on arXiv.
# sys.executable is the way to preferred way execute the current Python interpreter in a streamlit app,
# see streamlit docs: https://docs.streamlit.io/knowledge-base/deploy/invoking-python-subprocess-deployed-streamlit-app
arxiv_mcp = MCPServerStdio(
    command = f"{sys.executable}",
    args = ["arxiv_mcp.py"],
)

arxiv_agent = Agent('openai:gpt-4o', toolsets = [arxiv_mcp])


################
## Deps and Tools
################

# we also define additional tools and dependencies (including for state management) for the agent.
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

# this tool is in addition to those provided by the MCP server.
@arxiv_agent.tool
async def add_to_library(ctx: RunContext[Library], article: str) -> str:
    """Add a given article to the library."""
    ctx.deps.add(article)
    return f"Article added. Current library size: {len(ctx.deps.state.library)}"


################
## Streamlit Sidebar
################

# a function to render the agent's sidebar in Streamlit.
# It will be passed the deps object, which contains the agent's state and any other dependencies. 
async def arxiv_sidebar(deps):
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
    "arXiv Bot": AgentConfig(agent = arxiv_agent,
                             deps = Library(),
                             sidebar_func = arxiv_sidebar,
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


app_config = AppConfig(sidebar_collapsed= False,
                       page_icon= "📖",
                       page_title= "arXiv Bot",
                       rendering_functions= [render_df, show_warning],)


# This tool makes use of the `render_df` function to display the library as a DataFrame in the chat.
@arxiv_agent.tool
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

