""" widge label - sentiment labelling widget """

from IPython.display import display, Markdown
from ipywidgets import interact, interactive, fixed, interact_manual, VBox, Layout
import ipywidgets as widgets
import numpy as np
import pandas as pd
import re
import os.path


class LabellingClient():
    """ Client that creates labelling widget.
    """

    def __init__(self, text_file, label_file,
                 text_key,
                 unique_id_key= id,
                 additional_cols_names = [],
                 tags_file="default_tags.csv",
                 start_index=None,
                 emphasis_regex="",
                 emphasis_color="blue",
                 auto_save=False,
                 width = "650px",
                 hansard_tags = False,
                 sentiment_options=['positive', 'slightly positive', 'slightly negative', 'negative']):
        """ Setting up the client.
        
        Parameters
        ----------
        text_file : Path to the text file to be annotated
        label_file : Path to the label file where new labels should be appended and saved. 
            Will be create if it doesn't exist yet
        text_key : Name of text column in text_file
        unique_id_key : Column of unique id for each text, by default "id"
        additional_cols_names : list of additional columns of the text data to show, by default []
        tags_file : Path to the tags file where tags should be saved, optional, by default "default_tags.csv"
        start_index : [type], optional; index of first text to label, by default None
        emphasis_regex : Regular expression to highlight, optional, by default ""
        emphasis_color : Color of highlight, optional, by default "blue"
        auto_save : bool wether to save automatically, optional, by default False
        width : string of pixel width of the widget, optional, by default "650px"
        hansard_tags: bool whether to use special tags for usage with hasard data set (e.g. is_in_gov)
        sentiment_options: list of sentiments to choose from
        """
        
        self.label_file = label_file
        self.tags_file = tags_file
        self.text_key = text_key
        self.uid_key = unique_id_key
        self.additional_cols_names = additional_cols_names
        self.current_text_index = start_index
        self.emphasis_regex = emphasis_regex
        self.emphasis_color = emphasis_color
        self.auto_save = auto_save
        self.hansard_tags = hansard_tags
        self.sentiment_options = sentiment_options
        
        self.width = width
        self.sentiment_button_width = "100px"
        self.nav_button_width = "125px"
        
        self.out_nav_input = widgets.Output()
        self.out_sent_input = widgets.Output()
        self.out_text = widgets.Output()
        self.out_sent_data = widgets.Output()
        self.out_tags_data = widgets.Output()
        self.out_debug = widgets.Output()
        
        self.load_text_data(text_file)
        
        self.sentiment_df = pd.DataFrame(columns=[self.uid_key,"sentiment"])
        #self.relevancy_df = pd.DataFrame(columns=[self.uid_key,"relevancy"])
        self.tags_df = pd.DataFrame(columns=[self.uid_key,"tags"])
        self.return_later_df = pd.DataFrame(columns=[self.uid_key,"return_later"])
        
        # checking if label file already exists
        if os.path.isfile(self.label_file):
            self.load_label_data(self.label_file)
        else:
            with self.out_debug:
                print("No file {} found.".format(self.label_file)
                      + " Creating new file if saved or autosave activated.")
                
        if os.path.isfile(self.tags_file):
            self.load_tags_data(self.tags_file)
        else:
            with self.out_debug:
                print("No file {} found.".format(self.tags_file)
                      + " Creating new file if saved or autosave activated.")      
        
        # setting start index to last added to label file
        if start_index==None:
            if len(self.sentiment_df) > 0:
                last_uid_labelled = self.sentiment_df[self.uid_key].iloc[len(self.sentiment_df) - 1]
                self.current_text_index = self.find_text_index_by_uid(last_uid_labelled)
            else:
                self.current_text_index = 0
    
                
    # Button creation functions
    
    def create_nav_buttons(self):
        """ Generate the main navigation buttons of the widget
        """
        self._next_button = widgets.Button(description='',
                                    icon = 'arrow-right',
                                    tooltip = 'next text',
                                    layout=Layout(width=self.nav_button_width))
        self._prev_button = widgets.Button(description='',
                                    icon = 'arrow-left',
                                    tooltip = 'previous text',
                                    layout=Layout(width=self.nav_button_width))
        self._save_button = widgets.Button(description='Save Progress',
                                    icon = 'save',
                                    layout=Layout(width=self.nav_button_width))
        self._del_button = widgets.Button(description='Reset Current',
                                    icon = 'times',
                                    layout = Layout(width=self.nav_button_width))
                
        client_self = self
        
        def on_next_clicked(_):
            nonlocal client_self
            client_self.current_text_index += 1
            client_self.create_sent_select_subpanel_text_part()
            
        def on_prev_clicked(_):
            nonlocal client_self
            client_self.current_text_index -= 1
            client_self.create_sent_select_subpanel_text_part()
            
        def on_save_clicked(_):
            nonlocal client_self
            with client_self.out_debug:
                client_self.save_all()
                print("All data manually saved.")
                
        def on_del_clicked(_):
            nonlocal client_self
            client_self.sentiment_df = client_self.sentiment_df[client_self.sentiment_df[self.uid_key] != client_self.current_uid]
            if client_self.auto_save:
                client_self.save_label_data()
                with client_self.out_debug:
                    print ("Auto-saved data.")
            client_self.create_sent_select_subpanel_text_part()

        self._next_button.on_click(on_next_clicked)
        self._prev_button.on_click(on_prev_clicked)
        self._save_button.on_click(on_save_clicked)
        self._del_button.on_click(on_del_clicked)
        
        buttons = widgets.HBox([self._prev_button,self._next_button,self._save_button,self._del_button],
                               layout=Layout(width='100%'))
        
        return buttons
    
    def create_special_buttons(self):
        """Generate special buttons of the widget, "return later" and "not relevant".
        """
        return_later_button = widgets.ToggleButton(
            value = self.current_return_later,
            description = "return later",
            layout=Layout(width=self.nav_button_width)
        )
        
        not_rel_button = widgets.ToggleButton(
            value = self.current_not_rel,
            description = "not relevant",
            layout=Layout(width=self.nav_button_width)
        )
        
        special_buttons = widgets.HBox([return_later_button, not_rel_button])
        
        def f(client_self, return_later, not_rel):
            if return_later:
                client_self.add_tag(client_self.current_uid, "return_later")
            else:
                client_self.remove_tag(client_self.current_uid, "return_later")
                
            if not_rel:
                client_self.add_tag(client_self.current_uid, "not_rel")
            else:
                client_self.remove_tag(client_self.current_uid, "not_rel")
            
            client_self.update_data_panel()
            
        widgets.interactive_output(f, {'client_self': fixed(self),
                                       'return_later': return_later_button,
                                       'not_rel': not_rel_button})
        
        return special_buttons
    
    
    # Helper functions for tags
    
    def add_tag(self, uid, tag):
        with self.out_debug:
            if uid in self.tags_df[self.uid_key].values:
                tmp_tags = self.tags_df.loc[self.tags_df[self.uid_key] == uid].iloc[0].tags
                if not tag in tmp_tags:
                    tmp_tags.append(tag)
            else:
                tmp_tags = [tag]
                
            self.tags_df = self.tags_df[self.tags_df[self.uid_key] != uid]
            new_tags_row = {self.uid_key:uid, "tags":tmp_tags}
            self.tags_df = self.tags_df.append(new_tags_row, ignore_index=True)
            self.do_auto_save()
            
    def remove_tag(self, uid, tag):
        with self.out_debug:
            if uid in self.tags_df[self.uid_key].values:
                tmp_tags = self.tags_df.loc[self.tags_df[self.uid_key] == uid].iloc[0].tags
                if tag in tmp_tags:
                    tmp_tags.remove(tag)
                self.tags_df = self.tags_df[self.tags_df[self.uid_key] != uid]
                if not tmp_tags == []:
                    new_tags_row = {self.uid_key:uid, "tags":tmp_tags}
                    self.tags_df = self.tags_df.append(new_tags_row, ignore_index=True)
            
            self.do_auto_save()
            
    def check_tag(self, uid, tag):
        if uid in self.tags_df[self.uid_key].values:
                tmp_tags = self.tags_df.loc[self.tags_df[self.uid_key] == uid].iloc[0].tags
                if tag in tmp_tags:
                    return True
        
        return False
        
    def set_current_variables(self, text_index):
        """Fill variables from database by text index.
        
        Parameters
        ----------
        text_index : Index of text in database
        """
        
        self.current_text_index = text_index
        self.current_uid = self.text_df.iloc[text_index][self.uid_key]
        
        # Set special tags and variables for hansard data
        if self.hansard_tags == True:
            self.current_in_gov = self.check_is_in_gov(text_index)
        
        self.current_add_cols = {}
        for column in self.additional_cols_names:
            self.current_add_cols[column] = self.text_df.iloc[text_index][column]
        
        self.current_return_later = self.check_tag(self.current_uid, "return_later")
        self.current_not_rel = self.check_tag(self.current_uid, "not_rel")
        
        already_labeled = self.current_uid in self.sentiment_df[self.uid_key].values
        if already_labeled:
            self.current_sentiment = self.sentiment_df.loc[self.sentiment_df[self.uid_key] == self.current_uid].sentiment.values
        else:
            self.current_sentiment = None
            
            
    # Widget creation (seperated into the different widget parts)
        
    def create_sent_select_subpanel_text_part(self):
        """Create subpanel for text (with labelling options).
        """
        
        self.set_current_variables(self.current_text_index)
        
        def f(client_self, sentiment, current_uid):
            tmp_uid = client_self.text_df.iloc[client_self.current_text_index][self.uid_key]
                
            if sentiment:
                new_sent_row = {self.uid_key:tmp_uid, "sentiment":sentiment}
                if tmp_uid in client_self.sentiment_df[client_self.uid_key].values:
                    client_self.sentiment_df = client_self.sentiment_df[client_self.sentiment_df[client_self.uid_key] != tmp_uid]
                client_self.sentiment_df = client_self.sentiment_df.append(new_sent_row, ignore_index=True)
                if client_self.auto_save:
                    client_self.save_label_data()
                    with client_self.out_debug:
                        print ("Auto-saved data.")
            
            client_self.update_data_panel()
                
                    
            return sentiment
            
        special_buttons = self.create_special_buttons()
        
        sentiment_buttons = widgets.ToggleButtons(
                value = self.current_sentiment,
                options=self.sentiment_options,
                description = "Sentiment:",
                continuous_update = True,
                style = {"button_width": self.sentiment_button_width, "description_width": "66px"}
                )
        
        text_widget = interactive(f,
                                  sentiment=sentiment_buttons,
                                  client_self = fixed(self),
                                  current_uid = fixed(self.current_uid)
                                  )
        
        def create_tag_str():
            
            def add_tag_format(tag, color="darkgrey"):
                return " <span style=\"color:white;border-radius:5px;background:{};padding: 1px 3px 1px 3px;\">".format(color) + tag + "</span>"
            
            tag_str = ""
            for column_name, column_value in self.current_add_cols.items():
                tag = column_name + ": " + str(column_value)
                tag = add_tag_format(tag)
                tag_str += tag
                
            if self.hansard_tags == True:
                
                if self.current_in_gov:
                    gov_tag_col = "grey"
                else:
                    gov_tag_col = "darkgrey"
                    
                tag_str += add_tag_format("is_in_gov: {}".format(self.current_in_gov), gov_tag_col)
                
                
                self.current_return_later = True
                if self.current_return_later:
                    tag_str += add_tag_format("return&nbsp;later", "navy")
                    
                self.current_relevant = True
                if not self.current_relevant:
                    tag_str += add_tag_format("not&nbsp;relevant", "navy")
            
            return tag_str
        
        tag_str = create_tag_str()
        
        # displaying all buttons that rely on current text
        self.out_sent_input.clear_output()
        with self.out_sent_input: 
            display(special_buttons)
            display(text_widget)
            display(Markdown(tag_str))
        
        # displaying actual text
        self.out_text.clear_output()        
        with self.out_text:
 
            text_tmp = self.text_df.iloc[self.current_text_index][self.text_key]
            text_tmp = self.add_match_formatting(text_tmp, self.emphasis_regex)
            display(Markdown(text_tmp))
            
        # Updates
        self.update_data_panel()
            
    def update_data_panel(self):
        """Updating the panel containing sentiment data.
        """
        
        self.out_sent_data.clear_output()   
        with self.out_sent_data:   
            display(self.sentiment_df)
            
        self.out_tags_data.clear_output()  
        with self.out_tags_data:
            display(self.tags_df)
    
    def create_sent_select_subpanel(self):
        """Creates the panel to select the sentiment.
        
        Returns
        -------
        subpanel for sentiment selection
        """
        buttons = self.create_nav_buttons()
        with self.out_nav_input:
            display(buttons)
        self.create_sent_select_subpanel_text_part()
        
        sent_select_text = widgets.VBox([self.out_text], layout=Layout(height="410px"))
        sentbuttons_text_box = widgets.VBox([self.out_sent_input, sent_select_text],
                                            layout=Layout(width='100%', height="100%"))
        sent_select_subpanel = widgets.VBox([self.out_nav_input, sentbuttons_text_box])
        
        return sent_select_subpanel
        
    def add_match_formatting(self, str, regex):
        """Add html color formatting to each matching substring of a given regex.
        
        Parameters
        ----------
        str : String to be formatted.
        regex : Regular expression to be matched.
        
        Returns
        -------
        Formatted string
        """
        match = re.search(regex, str)
        if match:
            if len(str[match.end():]) > 0:
                str = str[:match.start()] + "<span style=\"color:{}\">**".format(self.emphasis_color) + str[match.start():match.end()] + "**</span>" + self.add_match_formatting(str=str[match.end():], regex=regex)
        return str
    
    def create_widget(self):
        """Create the complete sentiment selection widget.
        
        Returns
        -------
        sentiment selection widget
        """
        sent_select_subpanel = self.create_sent_select_subpanel()     
        data_subpanel = widgets.VBox([self._save_button,
                                      widgets.HBox([self.out_sent_data, self.out_tags_data])])
        
        # creating the tab structure for widget
        tab = widgets.Tab(layout=Layout(width='99%', height='620px'))
        tab.children = [sent_select_subpanel, data_subpanel, self.out_debug]
        tab.set_title(0, 'Labelling')
        tab.set_title(1, 'Data Collected')
        tab.set_title(2, 'Debug')

        widget = widgets.VBox([tab],layout=Layout(width=self.width, height='700px'))
        
        return widget
    
    
    # Helper functions
        
    def save_label_data(self, file = None):
        if not file:
            file = self.label_file
            
        self.sentiment_df.to_csv(file, index=False)
        
    def load_label_data(self, file):
        self.sentiment_df = pd.read_csv(file, index_col=False)
        
    def save_tags_data(self, file = None):
        if not file:
            file = self.tags_file
            
        self.tags_df.to_csv(file, index=False)
        
    def load_tags_data(self, file):
        self.tags_df = pd.read_csv(file, index_col=False)
        
    def load_text_data(self, file):
        self.text_df = pd.read_csv(file)
        
    def do_auto_save(self):
        if self.auto_save:
            self.save_all()
            
    def save_all(self):
        self.save_label_data()
        self.save_tags_data()
        
        with self.out_debug:
            print("Auto-saved all data.")
        
    def find_text_index_by_uid(self,uid):
        index = self.text_df.index[self.text_df[self.uid_key] == uid].tolist()[0]
        return index
    
    def check_is_in_gov(self, text_index):
        post_name = str(self.text_df.iloc[text_index]["post_name"])
        opp_regex = "(?i)shadow|opposition"
        opp_post = bool(re.search(opp_regex, post_name))
        if post_name != "nan" and not opp_post:
            in_gov = True
        else:
            in_gov = False
            
        return in_gov
        
