function [synch_Pol_tim] = find_PolTS_SynchCh(channelID,synch_cha,synch_tim,synch_pol,polarity)
% For a given channel, find the time at which 
id_ch = find(synch_cha==channelID);
synch_tim_ch = synch_tim(id_ch);
synch_Pol_tim = synch_tim_ch(find(synch_pol(id_ch)==polarity));

end

